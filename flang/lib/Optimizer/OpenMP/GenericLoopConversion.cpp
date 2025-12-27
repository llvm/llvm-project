//===- GenericLoopConversion.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Support/OpenMP-utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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
    : public mlir::OpConversionPattern<mlir::omp::LoopOp> {
public:
  enum class GenericLoopCombinedInfo { Standalone, TeamsLoop, ParallelLoop };

  using mlir::OpConversionPattern<mlir::omp::LoopOp>::OpConversionPattern;

  explicit GenericLoopConversionPattern(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<mlir::omp::LoopOp>{ctx} {
    // Enable rewrite recursion to make sure nested `loop` directives are
    // handled.
    this->setHasBoundedRewriteRecursion(true);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::omp::LoopOp loopOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(mlir::succeeded(checkLoopConversionSupportStatus(loopOp)));

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
        auto teamsOp = llvm::cast<mlir::omp::TeamsOp>(loopOp->getParentOp());
        auto teamsBlockArgIface =
            llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*teamsOp);
        auto loopBlockArgIface =
            llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*loopOp);

        for (unsigned i = 0; i < loopBlockArgIface.numReductionBlockArgs();
             ++i) {
          mlir::BlockArgument loopRedBlockArg =
              loopBlockArgIface.getReductionBlockArgs()[i];
          mlir::BlockArgument teamsRedBlockArg =
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
    return mlir::success();
  }

  static mlir::LogicalResult
  checkLoopConversionSupportStatus(mlir::omp::LoopOp loopOp) {
    auto todo = [&loopOp](mlir::StringRef clauseName) {
      return loopOp.emitError()
             << "not yet implemented: Unhandled clause " << clauseName << " in "
             << loopOp->getName() << " operation";
    };

    if (loopOp.getOrder())
      return todo("order");

    return mlir::success();
  }

private:
  static GenericLoopCombinedInfo
  findGenericLoopCombineInfo(mlir::omp::LoopOp loopOp) {
    mlir::Operation *parentOp = loopOp->getParentOp();
    GenericLoopCombinedInfo result = GenericLoopCombinedInfo::Standalone;

    if (auto teamsOp = mlir::dyn_cast_if_present<mlir::omp::TeamsOp>(parentOp))
      result = GenericLoopCombinedInfo::TeamsLoop;

    if (auto parallelOp =
            mlir::dyn_cast_if_present<mlir::omp::ParallelOp>(parentOp))
      result = GenericLoopCombinedInfo::ParallelLoop;

    return result;
  }

  /// Checks whether a `teams loop` construct can be rewriten to `teams
  /// distribute parallel do` or it has to be converted to `teams distribute`.
  ///
  /// This checks similar constrains to what is checked by `TeamsLoopChecker` in
  /// SemaOpenMP.cpp in clang.
  static bool teamsLoopCanBeParallelFor(mlir::omp::LoopOp loopOp) {
    bool canBeParallelFor =
        !loopOp
             .walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *nestedOp) {
               if (nestedOp == loopOp)
                 return mlir::WalkResult::advance();

               if (auto nestedLoopOp =
                       mlir::dyn_cast<mlir::omp::LoopOp>(nestedOp)) {
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
                         mlir::omp::ClauseBindKind::Parallel)
                   return mlir::WalkResult::interrupt();

                 if (combinedInfo == GenericLoopCombinedInfo::ParallelLoop)
                   return mlir::WalkResult::interrupt();

               } else if (auto callOp =
                              mlir::dyn_cast<mlir::CallOpInterface>(nestedOp)) {
                 // Calls to non-OpenMP API runtime functions inhibits
                 // transformation to `teams distribute parallel do` since the
                 // called functions might have nested parallelism themselves.
                 bool isOpenMPAPI = false;
                 mlir::CallInterfaceCallable callable =
                     callOp.getCallableForCallee();

                 if (auto callableSymRef =
                         mlir::dyn_cast<mlir::SymbolRefAttr>(callable))
                   isOpenMPAPI =
                       callableSymRef.getRootReference().strref().starts_with(
                           "omp_");

                 if (!isOpenMPAPI)
                   return mlir::WalkResult::interrupt();
               }

               return mlir::WalkResult::advance();
             })
             .wasInterrupted();

    return canBeParallelFor;
  }

  void rewriteStandaloneLoop(mlir::omp::LoopOp loopOp,
                             mlir::ConversionPatternRewriter &rewriter) const {
    using namespace mlir::omp;
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
  void rewriteToSimdLoop(mlir::omp::LoopOp loopOp,
                         mlir::ConversionPatternRewriter &rewriter) const {
    loopOp.emitWarning(
        "Detected standalone OpenMP `loop` directive with thread binding, "
        "the associated loop will be rewritten to `simd`.");
    rewriteToSingleWrapperOp<mlir::omp::SimdOp, mlir::omp::SimdOperands>(
        loopOp, rewriter);
  }

  void rewriteToDistribute(mlir::omp::LoopOp loopOp,
                           mlir::ConversionPatternRewriter &rewriter) const {
    assert(loopOp.getReductionVars().empty());
    rewriteToSingleWrapperOp<mlir::omp::DistributeOp,
                             mlir::omp::DistributeOperands>(loopOp, rewriter);
  }

  void rewriteToWsloop(mlir::omp::LoopOp loopOp,
                       mlir::ConversionPatternRewriter &rewriter) const {
    rewriteToSingleWrapperOp<mlir::omp::WsloopOp, mlir::omp::WsloopOperands>(
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
  rewriteToSingleWrapperOp(mlir::omp::LoopOp loopOp,
                           mlir::ConversionPatternRewriter &rewriter) const {
    OpOperandsTy clauseOps;
    clauseOps.privateVars = loopOp.getPrivateVars();

    auto privateSyms = loopOp.getPrivateSyms();
    if (privateSyms)
      clauseOps.privateSyms.assign(privateSyms->begin(), privateSyms->end());

    Fortran::common::openmp::EntryBlockArgs args;
    args.priv.vars = clauseOps.privateVars;

    if constexpr (!std::is_same_v<OpOperandsTy,
                                  mlir::omp::DistributeOperands>) {
      populateReductionClauseOps(loopOp, clauseOps);
      args.reduction.vars = clauseOps.reductionVars;
    }

    auto wrapperOp = OpTy::create(rewriter, loopOp.getLoc(), clauseOps);
    mlir::Block *opBlock = genEntryBlock(rewriter, args, wrapperOp.getRegion());

    mlir::IRMapping mapper;
    mlir::Block &loopBlock = *loopOp.getRegion().begin();

    for (auto [loopOpArg, opArg] :
         llvm::zip_equal(loopBlock.getArguments(), opBlock->getArguments()))
      mapper.map(loopOpArg, opArg);

    rewriter.clone(*loopOp.begin(), mapper);
  }

  void rewriteToDistributeParallelDo(
      mlir::omp::LoopOp loopOp,
      mlir::ConversionPatternRewriter &rewriter) const {
    mlir::omp::ParallelOperands parallelClauseOps;
    parallelClauseOps.privateVars = loopOp.getPrivateVars();

    auto privateSyms = loopOp.getPrivateSyms();
    if (privateSyms)
      parallelClauseOps.privateSyms.assign(privateSyms->begin(),
                                           privateSyms->end());

    Fortran::common::openmp::EntryBlockArgs parallelArgs;
    parallelArgs.priv.vars = parallelClauseOps.privateVars;

    auto parallelOp = mlir::omp::ParallelOp::create(rewriter, loopOp.getLoc(),
                                                    parallelClauseOps);
    genEntryBlock(rewriter, parallelArgs, parallelOp.getRegion());
    parallelOp.setComposite(true);
    rewriter.setInsertionPoint(
        mlir::omp::TerminatorOp::create(rewriter, loopOp.getLoc()));

    mlir::omp::DistributeOperands distributeClauseOps;
    auto distributeOp = mlir::omp::DistributeOp::create(
        rewriter, loopOp.getLoc(), distributeClauseOps);
    distributeOp.setComposite(true);
    rewriter.createBlock(&distributeOp.getRegion());

    mlir::omp::WsloopOperands wsloopClauseOps;
    populateReductionClauseOps(loopOp, wsloopClauseOps);
    Fortran::common::openmp::EntryBlockArgs wsloopArgs;
    wsloopArgs.reduction.vars = wsloopClauseOps.reductionVars;

    auto wsloopOp =
        mlir::omp::WsloopOp::create(rewriter, loopOp.getLoc(), wsloopClauseOps);
    wsloopOp.setComposite(true);
    genEntryBlock(rewriter, wsloopArgs, wsloopOp.getRegion());

    mlir::IRMapping mapper;

    auto loopBlockInterface =
        llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*loopOp);
    auto parallelBlockInterface =
        llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*parallelOp);
    auto wsloopBlockInterface =
        llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*wsloopOp);

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
  populateReductionClauseOps(mlir::omp::LoopOp loopOp,
                             mlir::omp::ReductionClauseOps &clauseOps) const {
    clauseOps.reductionMod = loopOp.getReductionModAttr();
    clauseOps.reductionVars = loopOp.getReductionVars();

    std::optional<mlir::ArrayAttr> reductionSyms = loopOp.getReductionSyms();
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
    : public mlir::OpConversionPattern<mlir::omp::TeamsOp> {
public:
  using mlir::OpConversionPattern<mlir::omp::TeamsOp>::OpConversionPattern;

  static mlir::omp::LoopOp
  tryToFindNestedLoopWithReduction(mlir::omp::TeamsOp teamsOp) {
    if (teamsOp.getRegion().getBlocks().size() != 1)
      return nullptr;

    mlir::Block &teamsBlock = *teamsOp.getRegion().begin();
    auto loopOpIter = llvm::find_if(teamsBlock, [](mlir::Operation &op) {
      auto nestedLoopOp = llvm::dyn_cast<mlir::omp::LoopOp>(&op);

      if (!nestedLoopOp)
        return false;

      return !nestedLoopOp.getReductionVars().empty();
    });

    if (loopOpIter == teamsBlock.end())
      return nullptr;

    // TODO return error if more than one loop op is nested. We need to
    // coalesce reductions in this case.
    return llvm::cast<mlir::omp::LoopOp>(loopOpIter);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::omp::TeamsOp teamsOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::omp::LoopOp nestedLoopOp = tryToFindNestedLoopWithReduction(teamsOp);

    rewriter.modifyOpInPlace(teamsOp, [&]() {
      teamsOp.setReductionMod(nestedLoopOp.getReductionMod());
      teamsOp.getReductionVarsMutable().assign(nestedLoopOp.getReductionVars());
      teamsOp.setReductionByref(nestedLoopOp.getReductionByref());
      teamsOp.setReductionSymsAttr(nestedLoopOp.getReductionSymsAttr());

      auto blockArgIface =
          llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*teamsOp);
      unsigned reductionArgsStart = blockArgIface.getPrivateBlockArgsStart() +
                                    blockArgIface.numPrivateBlockArgs();
      llvm::SmallVector<mlir::Value> newLoopOpReductionOperands;

      for (auto [idx, reductionVar] :
           llvm::enumerate(nestedLoopOp.getReductionVars())) {
        mlir::BlockArgument newTeamsOpReductionBlockArg =
            teamsOp.getRegion().insertArgument(reductionArgsStart + idx,
                                               reductionVar.getType(),
                                               reductionVar.getLoc());
        newLoopOpReductionOperands.push_back(newTeamsOpReductionBlockArg);
      }

      nestedLoopOp.getReductionVarsMutable().assign(newLoopOpReductionOperands);
    });

    return mlir::success();
  }
};

class GenericLoopConversionPass
    : public flangomp::impl::GenericLoopConversionPassBase<
          GenericLoopConversionPass> {
public:
  GenericLoopConversionPass() = default;

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    if (func.isDeclaration())
      return;

    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<ReductionsHoistingPattern, GenericLoopConversionPattern>(
        context);
    mlir::ConversionTarget target(*context);

    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });

    target.addDynamicallyLegalOp<mlir::omp::TeamsOp>(
        [](mlir::omp::TeamsOp teamsOp) {
          // If teamsOp's reductions are already populated, then the op is
          // legal. Additionally, the op is legal if it does not nest a LoopOp
          // with reductions.
          return !teamsOp.getReductionVars().empty() ||
                 ReductionsHoistingPattern::tryToFindNestedLoopWithReduction(
                     teamsOp) == nullptr;
        });

    target.addDynamicallyLegalOp<mlir::omp::LoopOp>(
        [](mlir::omp::LoopOp loopOp) {
          return mlir::failed(
              GenericLoopConversionPattern::checkLoopConversionSupportStatus(
                  loopOp));
        });

    mlir::ConversionConfig config;
    config.allowPatternRollback = false;
    if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                               std::move(patterns), config))) {
      mlir::emitError(func.getLoc(), "error in converting `omp.loop` op");
      signalPassFailure();
    }
  }
};
} // namespace
