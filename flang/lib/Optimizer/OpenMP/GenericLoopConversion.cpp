//===- GenericLoopConversion.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Support/OpenMP-utils.h"

#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "aiir/IR/IRMapping.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"

#include <memory>
#include <optional>
#include <type_traits>

namespace flangomp {
#define GEN_PASS_DEF_GENERICLOOPCONVERSIONPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

namespace {

/// A conversion pattern to handle various combined forms of `omp.loop`. For how
/// combined/composite directive are handled see:
/// https://discourse.llvm.org/t/rfc-representing-combined-composite-constructs-in-the-openmp-dialect/76986.
class GenericLoopConversionPattern
    : public aiir::OpConversionPattern<aiir::omp::LoopOp> {
public:
  enum class GenericLoopCombinedInfo { Standalone, TeamsLoop, ParallelLoop };

  using aiir::OpConversionPattern<aiir::omp::LoopOp>::OpConversionPattern;

  explicit GenericLoopConversionPattern(aiir::AIIRContext *ctx)
      : aiir::OpConversionPattern<aiir::omp::LoopOp>{ctx} {
    // Enable rewrite recursion to make sure nested `loop` directives are
    // handled.
    this->setHasBoundedRewriteRecursion(true);
  }

  aiir::LogicalResult
  matchAndRewrite(aiir::omp::LoopOp loopOp, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    assert(aiir::succeeded(checkLoopConversionSupportStatus(loopOp)));

    GenericLoopCombinedInfo combinedInfo = findGenericLoopCombineInfo(loopOp);

    switch (combinedInfo) {
    case GenericLoopCombinedInfo::Standalone:
      rewriteStandaloneLoop(loopOp, rewriter);
      break;
    case GenericLoopCombinedInfo::ParallelLoop:
      rewriteToWsloop(loopOp, rewriter);
      break;
    case GenericLoopCombinedInfo::TeamsLoop:
      if (teamsLoopCanBeParallelFor(loopOp)) {
        rewriteToDistributeParallelDo(loopOp, rewriter);
      } else {
        auto teamsOp = llvm::cast<aiir::omp::TeamsOp>(loopOp->getParentOp());
        auto teamsBlockArgIface =
            llvm::cast<aiir::omp::BlockArgOpenMPOpInterface>(*teamsOp);
        auto loopBlockArgIface =
            llvm::cast<aiir::omp::BlockArgOpenMPOpInterface>(*loopOp);

        for (unsigned i = 0; i < loopBlockArgIface.numReductionBlockArgs();
             ++i) {
          aiir::BlockArgument loopRedBlockArg =
              loopBlockArgIface.getReductionBlockArgs()[i];
          aiir::BlockArgument teamsRedBlockArg =
              teamsBlockArgIface.getReductionBlockArgs()[i];
          rewriter.replaceAllUsesWith(loopRedBlockArg, teamsRedBlockArg);
        }

        for (unsigned i = 0; i < loopBlockArgIface.numReductionBlockArgs();
             ++i) {
          loopOp.getRegion().eraseArgument(
              loopBlockArgIface.getReductionBlockArgsStart());
        }

        loopOp.removeReductionModAttr();
        loopOp.getReductionVarsMutable().clear();
        loopOp.removeReductionByrefAttr();
        loopOp.removeReductionSymsAttr();

        rewriteToDistribute(loopOp, rewriter);
      }

      break;
    }

    rewriter.eraseOp(loopOp);
    return aiir::success();
  }

  static aiir::LogicalResult
  checkLoopConversionSupportStatus(aiir::omp::LoopOp loopOp) {
    auto todo = [&loopOp](aiir::StringRef clauseName) {
      return loopOp.emitError()
             << "not yet implemented: Unhandled clause " << clauseName << " in "
             << loopOp->getName() << " operation";
    };

    if (loopOp.getOrder())
      return todo("order");

    return aiir::success();
  }

private:
  static GenericLoopCombinedInfo
  findGenericLoopCombineInfo(aiir::omp::LoopOp loopOp) {
    aiir::Operation *parentOp = loopOp->getParentOp();
    GenericLoopCombinedInfo result = GenericLoopCombinedInfo::Standalone;

    if (auto teamsOp = aiir::dyn_cast_if_present<aiir::omp::TeamsOp>(parentOp))
      result = GenericLoopCombinedInfo::TeamsLoop;

    if (auto parallelOp =
            aiir::dyn_cast_if_present<aiir::omp::ParallelOp>(parentOp))
      result = GenericLoopCombinedInfo::ParallelLoop;

    return result;
  }

  /// Checks whether a `teams loop` construct can be rewriten to `teams
  /// distribute parallel do` or it has to be converted to `teams distribute`.
  ///
  /// This checks similar constrains to what is checked by `TeamsLoopChecker` in
  /// SemaOpenMP.cpp in clang.
  static bool teamsLoopCanBeParallelFor(aiir::omp::LoopOp loopOp) {
    bool canBeParallelFor =
        !loopOp
             .walk<aiir::WalkOrder::PreOrder>([&](aiir::Operation *nestedOp) {
               if (nestedOp == loopOp)
                 return aiir::WalkResult::advance();

               if (auto nestedLoopOp =
                       aiir::dyn_cast<aiir::omp::LoopOp>(nestedOp)) {
                 GenericLoopCombinedInfo combinedInfo =
                     findGenericLoopCombineInfo(nestedLoopOp);

                 // Worksharing loops cannot be nested inside each other.
                 // Therefore, if the current `loop` directive nests another
                 // `loop` whose `bind` modifier is `parallel`, this `loop`
                 // directive cannot be mapped to `distribute parallel for`
                 // but rather only to `distribute`.
                 if (combinedInfo == GenericLoopCombinedInfo::Standalone &&
                     nestedLoopOp.getBindKind() &&
                     *nestedLoopOp.getBindKind() ==
                         aiir::omp::ClauseBindKind::Parallel)
                   return aiir::WalkResult::interrupt();

                 if (combinedInfo == GenericLoopCombinedInfo::ParallelLoop)
                   return aiir::WalkResult::interrupt();

               } else if (auto callOp =
                              aiir::dyn_cast<aiir::CallOpInterface>(nestedOp)) {
                 // Calls to non-OpenMP API runtime functions inhibits
                 // transformation to `teams distribute parallel do` since the
                 // called functions might have nested parallelism themselves.
                 bool isOpenMPAPI = false;
                 aiir::CallInterfaceCallable callable =
                     callOp.getCallableForCallee();

                 if (auto callableSymRef =
                         aiir::dyn_cast<aiir::SymbolRefAttr>(callable))
                   isOpenMPAPI =
                       callableSymRef.getRootReference().strref().starts_with(
                           "omp_");

                 if (!isOpenMPAPI)
                   return aiir::WalkResult::interrupt();
               }

               return aiir::WalkResult::advance();
             })
             .wasInterrupted();

    return canBeParallelFor;
  }

  void rewriteStandaloneLoop(aiir::omp::LoopOp loopOp,
                             aiir::ConversionPatternRewriter &rewriter) const {
    using namespace aiir::omp;
    std::optional<ClauseBindKind> bindKind = loopOp.getBindKind();

    if (!bindKind.has_value())
      return rewriteToSimdLoop(loopOp, rewriter);

    switch (*loopOp.getBindKind()) {
    case ClauseBindKind::Parallel:
      return rewriteToWsloop(loopOp, rewriter);
    case ClauseBindKind::Teams:
      return rewriteToDistribute(loopOp, rewriter);
    case ClauseBindKind::Thread:
      return rewriteToSimdLoop(loopOp, rewriter);
    }
  }

  /// Rewrites standalone `loop` (without `bind` clause or with
  /// `bind(parallel)`) directives to equivalent `simd` constructs.
  ///
  /// The reasoning behind this decision is that according to the spec (version
  /// 5.2, section 11.7.1):
  ///
  /// "If the bind clause is not specified on a construct for which it may be
  /// specified and the construct is closely nested inside a teams or parallel
  /// construct, the effect is as if binding is teams or parallel. If none of
  /// those conditions hold, the binding region is not defined."
  ///
  /// which means that standalone `loop` directives have undefined binding
  /// region. Moreover, the spec says (in the next paragraph):
  ///
  /// "The specified binding region determines the binding thread set.
  /// Specifically, if the binding region is a teams region, then the binding
  /// thread set is the set of initial threads that are executing that region
  /// while if the binding region is a parallel region, then the binding thread
  /// set is the team of threads that are executing that region. If the binding
  /// region is not defined, then the binding thread set is the encountering
  /// thread."
  ///
  /// which means that the binding thread set for a standalone `loop` directive
  /// is only the encountering thread.
  ///
  /// Since the encountering thread is the binding thread (set) for a
  /// standalone `loop` directive, the best we can do in such case is to "simd"
  /// the directive.
  void rewriteToSimdLoop(aiir::omp::LoopOp loopOp,
                         aiir::ConversionPatternRewriter &rewriter) const {
    loopOp.emitWarning(
        "Detected standalone OpenMP `loop` directive with thread binding, "
        "the associated loop will be rewritten to `simd`.");
    rewriteToSingleWrapperOp<aiir::omp::SimdOp, aiir::omp::SimdOperands>(
        loopOp, rewriter);
  }

  void rewriteToDistribute(aiir::omp::LoopOp loopOp,
                           aiir::ConversionPatternRewriter &rewriter) const {
    assert(loopOp.getReductionVars().empty());
    rewriteToSingleWrapperOp<aiir::omp::DistributeOp,
                             aiir::omp::DistributeOperands>(loopOp, rewriter);
  }

  void rewriteToWsloop(aiir::omp::LoopOp loopOp,
                       aiir::ConversionPatternRewriter &rewriter) const {
    rewriteToSingleWrapperOp<aiir::omp::WsloopOp, aiir::omp::WsloopOperands>(
        loopOp, rewriter);
  }

  // TODO Suggestion by Sergio: tag auto-generated operations for constructs
  // that weren't part of the original program, that would be useful
  // information for debugging purposes later on. This new attribute could be
  // used for `omp.loop`, but also for `do concurrent` transformations,
  // `workshare`, `workdistribute`, etc. The tag could be used for all kinds of
  // auto-generated operations using a dialect attribute (named something like
  // `omp.origin` or `omp.derived`) and perhaps hold the name of the operation
  // it was derived from, the reason it was transformed or something like that
  // we could use when emitting any messages related to it later on.
  template <typename OpTy, typename OpOperandsTy>
  void
  rewriteToSingleWrapperOp(aiir::omp::LoopOp loopOp,
                           aiir::ConversionPatternRewriter &rewriter) const {
    OpOperandsTy clauseOps;
    clauseOps.privateVars = loopOp.getPrivateVars();

    auto privateSyms = loopOp.getPrivateSyms();
    if (privateSyms)
      clauseOps.privateSyms.assign(privateSyms->begin(), privateSyms->end());

    Fortran::common::openmp::EntryBlockArgs args;
    args.priv.vars = clauseOps.privateVars;

    if constexpr (!std::is_same_v<OpOperandsTy,
                                  aiir::omp::DistributeOperands>) {
      populateReductionClauseOps(loopOp, clauseOps);
      args.reduction.vars = clauseOps.reductionVars;
    }

    auto wrapperOp = OpTy::create(rewriter, loopOp.getLoc(), clauseOps);
    aiir::Block *opBlock = genEntryBlock(rewriter, args, wrapperOp.getRegion());

    aiir::IRMapping mapper;
    aiir::Block &loopBlock = *loopOp.getRegion().begin();

    for (auto [loopOpArg, opArg] :
         llvm::zip_equal(loopBlock.getArguments(), opBlock->getArguments()))
      mapper.map(loopOpArg, opArg);

    rewriter.clone(*loopOp.begin(), mapper);
  }

  void rewriteToDistributeParallelDo(
      aiir::omp::LoopOp loopOp,
      aiir::ConversionPatternRewriter &rewriter) const {
    aiir::omp::ParallelOperands parallelClauseOps;
    parallelClauseOps.privateVars = loopOp.getPrivateVars();

    auto privateSyms = loopOp.getPrivateSyms();
    if (privateSyms)
      parallelClauseOps.privateSyms.assign(privateSyms->begin(),
                                           privateSyms->end());

    Fortran::common::openmp::EntryBlockArgs parallelArgs;
    parallelArgs.priv.vars = parallelClauseOps.privateVars;

    auto parallelOp = aiir::omp::ParallelOp::create(rewriter, loopOp.getLoc(),
                                                    parallelClauseOps);
    genEntryBlock(rewriter, parallelArgs, parallelOp.getRegion());
    parallelOp.setComposite(true);
    rewriter.setInsertionPoint(
        aiir::omp::TerminatorOp::create(rewriter, loopOp.getLoc()));

    aiir::omp::DistributeOperands distributeClauseOps;
    auto distributeOp = aiir::omp::DistributeOp::create(
        rewriter, loopOp.getLoc(), distributeClauseOps);
    distributeOp.setComposite(true);
    rewriter.createBlock(&distributeOp.getRegion());

    aiir::omp::WsloopOperands wsloopClauseOps;
    populateReductionClauseOps(loopOp, wsloopClauseOps);
    Fortran::common::openmp::EntryBlockArgs wsloopArgs;
    wsloopArgs.reduction.vars = wsloopClauseOps.reductionVars;

    auto wsloopOp =
        aiir::omp::WsloopOp::create(rewriter, loopOp.getLoc(), wsloopClauseOps);
    wsloopOp.setComposite(true);
    genEntryBlock(rewriter, wsloopArgs, wsloopOp.getRegion());

    aiir::IRMapping mapper;

    auto loopBlockInterface =
        llvm::cast<aiir::omp::BlockArgOpenMPOpInterface>(*loopOp);
    auto parallelBlockInterface =
        llvm::cast<aiir::omp::BlockArgOpenMPOpInterface>(*parallelOp);
    auto wsloopBlockInterface =
        llvm::cast<aiir::omp::BlockArgOpenMPOpInterface>(*wsloopOp);

    for (auto [loopOpArg, parallelOpArg] :
         llvm::zip_equal(loopBlockInterface.getPrivateBlockArgs(),
                         parallelBlockInterface.getPrivateBlockArgs()))
      mapper.map(loopOpArg, parallelOpArg);

    for (auto [loopOpArg, wsloopOpArg] :
         llvm::zip_equal(loopBlockInterface.getReductionBlockArgs(),
                         wsloopBlockInterface.getReductionBlockArgs()))
      mapper.map(loopOpArg, wsloopOpArg);

    rewriter.clone(*loopOp.begin(), mapper);
  }

  void
  populateReductionClauseOps(aiir::omp::LoopOp loopOp,
                             aiir::omp::ReductionClauseOps &clauseOps) const {
    clauseOps.reductionMod = loopOp.getReductionModAttr();
    clauseOps.reductionVars = loopOp.getReductionVars();

    std::optional<aiir::ArrayAttr> reductionSyms = loopOp.getReductionSyms();
    if (reductionSyms)
      clauseOps.reductionSyms.assign(reductionSyms->begin(),
                                     reductionSyms->end());

    std::optional<llvm::ArrayRef<bool>> reductionByref =
        loopOp.getReductionByref();
    if (reductionByref)
      clauseOps.reductionByref.assign(reductionByref->begin(),
                                      reductionByref->end());
  }
};

/// According to the spec (v5.2, p340, 36):
///
/// ```
/// The effect of the reduction clause is as if it is applied to all leaf
/// constructs that permit the clause, except for the following constructs:
/// * ....
/// * The teams construct, when combined with the loop construct.
/// ```
///
/// Therefore, for a combined directive similar to: `!$omp teams loop
/// reduction(...)`, the earlier stages of the compiler assign the `reduction`
/// clauses only to the `loop` leaf and not to the `teams` leaf.
///
/// On the other hand, if we have a combined construct similar to: `!$omp teams
/// distribute parallel do`, the `reduction` clauses are assigned both to the
/// `teams` and the `do` leaves. We need to match this behavior when we convert
/// `teams` op with a nested `loop` op since the target set of constructs/ops
/// will be incorrect without moving the reductions up to the `teams` op as
/// well.
///
/// This pattern does exactly this. Given the following input:
/// ```
/// omp.teams {
///   omp.loop reduction(@red_sym %red_op -> %red_arg : !fir.ref<i32>) {
///     omp.loop_nest ... {
///       ...
///     }
///   }
/// }
/// ```
/// this pattern updates the `omp.teams` op in-place to:
/// ```
/// omp.teams reduction(@red_sym %red_op -> %teams_red_arg : !fir.ref<i32>) {
///   omp.loop reduction(@red_sym %teams_red_arg -> %red_arg : !fir.ref<i32>) {
///     omp.loop_nest ... {
///       ...
///     }
///   }
/// }
/// ```
///
/// Note the following:
/// * The nested `omp.loop` is not rewritten by this pattern, this happens
///   through `GenericLoopConversionPattern`.
/// * The reduction info are cloned from the nested `omp.loop` op to the parent
///   `omp.teams` op.
/// * The reduction operand of the `omp.loop` op is updated to be the **new**
///   reduction block argument of the `omp.teams` op.
class ReductionsHoistingPattern
    : public aiir::OpConversionPattern<aiir::omp::TeamsOp> {
public:
  using aiir::OpConversionPattern<aiir::omp::TeamsOp>::OpConversionPattern;

  static aiir::omp::LoopOp
  tryToFindNestedLoopWithReduction(aiir::omp::TeamsOp teamsOp) {
    if (teamsOp.getRegion().getBlocks().size() != 1)
      return nullptr;

    aiir::Block &teamsBlock = *teamsOp.getRegion().begin();
    auto loopOpIter = llvm::find_if(teamsBlock, [](aiir::Operation &op) {
      auto nestedLoopOp = llvm::dyn_cast<aiir::omp::LoopOp>(&op);

      if (!nestedLoopOp)
        return false;

      return !nestedLoopOp.getReductionVars().empty();
    });

    if (loopOpIter == teamsBlock.end())
      return nullptr;

    // TODO return error if more than one loop op is nested. We need to
    // coalesce reductions in this case.
    return llvm::cast<aiir::omp::LoopOp>(loopOpIter);
  }

  aiir::LogicalResult
  matchAndRewrite(aiir::omp::TeamsOp teamsOp, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::omp::LoopOp nestedLoopOp = tryToFindNestedLoopWithReduction(teamsOp);

    rewriter.modifyOpInPlace(teamsOp, [&]() {
      teamsOp.setReductionMod(nestedLoopOp.getReductionMod());
      teamsOp.getReductionVarsMutable().assign(nestedLoopOp.getReductionVars());
      teamsOp.setReductionByref(nestedLoopOp.getReductionByref());
      teamsOp.setReductionSymsAttr(nestedLoopOp.getReductionSymsAttr());

      auto blockArgIface =
          llvm::cast<aiir::omp::BlockArgOpenMPOpInterface>(*teamsOp);
      unsigned reductionArgsStart = blockArgIface.getPrivateBlockArgsStart() +
                                    blockArgIface.numPrivateBlockArgs();
      llvm::SmallVector<aiir::Value> newLoopOpReductionOperands;

      for (auto [idx, reductionVar] :
           llvm::enumerate(nestedLoopOp.getReductionVars())) {
        aiir::BlockArgument newTeamsOpReductionBlockArg =
            teamsOp.getRegion().insertArgument(reductionArgsStart + idx,
                                               reductionVar.getType(),
                                               reductionVar.getLoc());
        newLoopOpReductionOperands.push_back(newTeamsOpReductionBlockArg);
      }

      nestedLoopOp.getReductionVarsMutable().assign(newLoopOpReductionOperands);
    });

    return aiir::success();
  }
};

class GenericLoopConversionPass
    : public flangomp::impl::GenericLoopConversionPassBase<
          GenericLoopConversionPass> {
public:
  GenericLoopConversionPass() = default;

  void runOnOperation() override {
    aiir::func::FuncOp func = getOperation();

    if (func.isDeclaration())
      return;

    aiir::AIIRContext *context = &getContext();
    aiir::RewritePatternSet patterns(context);
    patterns.insert<ReductionsHoistingPattern, GenericLoopConversionPattern>(
        context);
    aiir::ConversionTarget target(*context);

    target.markUnknownOpDynamicallyLegal(
        [](aiir::Operation *) { return true; });

    target.addDynamicallyLegalOp<aiir::omp::TeamsOp>(
        [](aiir::omp::TeamsOp teamsOp) {
          // If teamsOp's reductions are already populated, then the op is
          // legal. Additionally, the op is legal if it does not nest a LoopOp
          // with reductions.
          return !teamsOp.getReductionVars().empty() ||
                 ReductionsHoistingPattern::tryToFindNestedLoopWithReduction(
                     teamsOp) == nullptr;
        });

    target.addDynamicallyLegalOp<aiir::omp::LoopOp>(
        [](aiir::omp::LoopOp loopOp) {
          return aiir::failed(
              GenericLoopConversionPattern::checkLoopConversionSupportStatus(
                  loopOp));
        });

    aiir::ConversionConfig config;
    config.allowPatternRollback = false;
    if (aiir::failed(aiir::applyFullConversion(getOperation(), target,
                                               std::move(patterns), config))) {
      aiir::emitError(func.getLoc(), "error in converting `omp.loop` op");
      signalPassFailure();
    }
  }
};
} // namespace
