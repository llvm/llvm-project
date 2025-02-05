//===- GenericLoopConversion.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/OpenMP-utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

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
      llvm_unreachable(
          "not yet implemented: Combined `parallel loop` directive");
      break;
    case GenericLoopCombinedInfo::TeamsLoop:
      rewriteToDistributeParallelDo(loopOp, rewriter);
      break;
    }

    rewriter.eraseOp(loopOp);
    return mlir::success();
  }

  static mlir::LogicalResult
  checkLoopConversionSupportStatus(mlir::omp::LoopOp loopOp) {
    GenericLoopCombinedInfo combinedInfo = findGenericLoopCombineInfo(loopOp);

    switch (combinedInfo) {
    case GenericLoopCombinedInfo::Standalone:
      break;
    case GenericLoopCombinedInfo::ParallelLoop:
      return loopOp.emitError(
          "not yet implemented: Combined `parallel loop` directive");
    case GenericLoopCombinedInfo::TeamsLoop:
      break;
    }

    auto todo = [&loopOp](mlir::StringRef clauseName) {
      return loopOp.emitError()
             << "not yet implemented: Unhandled clause " << clauseName << " in "
             << loopOp->getName() << " operation";
    };

    // For standalone directives, `bind` is already supported. Other combined
    // forms will be supported in a follow-up PR.
    if (combinedInfo != GenericLoopCombinedInfo::Standalone &&
        loopOp.getBindKind())
      return todo("bind");

    if (loopOp.getOrder())
      return todo("order");

    if (!loopOp.getReductionVars().empty())
      return todo("reduction");

    // TODO For `teams loop`, check similar constrains to what is checked
    // by `TeamsLoopChecker` in SemaOpenMP.cpp.
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
      return rewriteToDistrbute(loopOp, rewriter);
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

  void rewriteToDistrbute(mlir::omp::LoopOp loopOp,
                          mlir::ConversionPatternRewriter &rewriter) const {
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

    auto wrapperOp = rewriter.create<OpTy>(loopOp.getLoc(), clauseOps);
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

    auto parallelOp = rewriter.create<mlir::omp::ParallelOp>(loopOp.getLoc(),
                                                             parallelClauseOps);
    mlir::Block *parallelBlock =
        genEntryBlock(rewriter, parallelArgs, parallelOp.getRegion());
    parallelOp.setComposite(true);
    rewriter.setInsertionPoint(
        rewriter.create<mlir::omp::TerminatorOp>(loopOp.getLoc()));

    mlir::omp::DistributeOperands distributeClauseOps;
    auto distributeOp = rewriter.create<mlir::omp::DistributeOp>(
        loopOp.getLoc(), distributeClauseOps);
    distributeOp.setComposite(true);
    rewriter.createBlock(&distributeOp.getRegion());

    mlir::omp::WsloopOperands wsloopClauseOps;
    auto wsloopOp =
        rewriter.create<mlir::omp::WsloopOp>(loopOp.getLoc(), wsloopClauseOps);
    wsloopOp.setComposite(true);
    rewriter.createBlock(&wsloopOp.getRegion());

    mlir::IRMapping mapper;
    mlir::Block &loopBlock = *loopOp.getRegion().begin();

    for (auto [loopOpArg, parallelOpArg] : llvm::zip_equal(
             loopBlock.getArguments(), parallelBlock->getArguments()))
      mapper.map(loopOpArg, parallelOpArg);

    rewriter.clone(*loopOp.begin(), mapper);
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
    patterns.insert<GenericLoopConversionPattern>(context);
    mlir::ConversionTarget target(*context);

    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });
    target.addDynamicallyLegalOp<mlir::omp::LoopOp>(
        [](mlir::omp::LoopOp loopOp) {
          return mlir::failed(
              GenericLoopConversionPattern::checkLoopConversionSupportStatus(
                  loopOp));
        });

    if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                               std::move(patterns)))) {
      mlir::emitError(func.getLoc(), "error in converting `omp.loop` op");
      signalPassFailure();
    }
  }
};
} // namespace
