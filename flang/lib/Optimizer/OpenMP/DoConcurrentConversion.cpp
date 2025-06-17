//===- DoConcurrentConversion.cpp -- map `DO CONCURRENT` to OpenMP loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "flang/Optimizer/OpenMP/Utils.h"
#include "flang/Support/OpenMP-utils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

namespace flangomp {
#define GEN_PASS_DEF_DOCONCURRENTCONVERSIONPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "do-concurrent-conversion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {
namespace looputils {
/// Stores info needed about the induction/iteration variable for each `do
/// concurrent` in a loop nest.
struct InductionVariableInfo {
  InductionVariableInfo(fir::DoConcurrentLoopOp loop,
                        mlir::Value inductionVar) {
    populateInfo(loop, inductionVar);
  }
  /// The operation allocating memory for iteration variable.
  mlir::Operation *iterVarMemDef;
  /// the operation(s) updating the iteration variable with the current
  /// iteration number.
  llvm::SmallVector<mlir::Operation *, 2> indVarUpdateOps;

private:
  /// For the \p doLoop parameter, find the following:
  ///
  /// 1. The operation that declares its iteration variable or allocates memory
  /// for it. For example, give the following loop:
  /// ```
  ///   ...
  ///   %i:2 = hlfir.declare %0 {uniq_name = "_QFEi"} : ...
  ///   ...
  ///   fir.do_concurrent.loop (%ind_var) = (%lb) to (%ub) step (%s) {
  ///     %ind_var_conv = fir.convert %ind_var : (index) -> i32
  ///     fir.store %ind_var_conv to %i#1 : !fir.ref<i32>
  ///     ...
  ///   }
  /// ```
  ///
  /// This function sets the `iterVarMemDef` member to the `hlfir.declare` op
  /// for `%i`.
  ///
  /// 2. The operation(s) that update the loop's iteration variable from its
  /// induction variable. For the above example, the `indVarUpdateOps` is
  /// populated with the first 2 ops in the loop's body.
  ///
  /// Note: The current implementation is dependent on how flang emits loop
  /// bodies; which is sufficient for the current simple test/use cases. If this
  /// proves to be insufficient, this should be made more generic.
  void populateInfo(fir::DoConcurrentLoopOp loop, mlir::Value inductionVar) {
    mlir::Value result = nullptr;

    // Checks if a StoreOp is updating the memref of the loop's iteration
    // variable.
    auto isStoringIV = [&](fir::StoreOp storeOp) {
      // Direct store into the IV memref.
      if (storeOp.getValue() == inductionVar) {
        indVarUpdateOps.push_back(storeOp);
        return true;
      }

      // Indirect store into the IV memref.
      if (auto convertOp = mlir::dyn_cast<fir::ConvertOp>(
              storeOp.getValue().getDefiningOp())) {
        if (convertOp.getOperand() == inductionVar) {
          indVarUpdateOps.push_back(convertOp);
          indVarUpdateOps.push_back(storeOp);
          return true;
        }
      }

      return false;
    };

    for (mlir::Operation &op : loop) {
      if (auto storeOp = mlir::dyn_cast<fir::StoreOp>(op))
        if (isStoringIV(storeOp)) {
          result = storeOp.getMemref();
          break;
        }
    }

    assert(result != nullptr && result.getDefiningOp() != nullptr);
    iterVarMemDef = result.getDefiningOp();
  }
};

using InductionVariableInfos = llvm::SmallVector<InductionVariableInfo>;

/// Collects values that are local to a loop: "loop-local values". A loop-local
/// value is one that is used exclusively inside the loop but allocated outside
/// of it. This usually corresponds to temporary values that are used inside the
/// loop body for initialzing other variables for example.
///
/// See `flang/test/Transforms/DoConcurrent/locally_destroyed_temp.f90` for an
/// example of why we need this.
///
/// \param [in] doLoop - the loop within which the function searches for values
/// used exclusively inside.
///
/// \param [out] locals - the list of loop-local values detected for \p doLoop.
void collectLoopLocalValues(fir::DoConcurrentLoopOp loop,
                            llvm::SetVector<mlir::Value> &locals) {
  loop.walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands()) {
      if (locals.contains(operand))
        continue;

      bool isLocal = true;

      if (!mlir::isa_and_present<fir::AllocaOp>(operand.getDefiningOp()))
        continue;

      // Values defined inside the loop are not interesting since they do not
      // need to be localized.
      if (loop->isAncestor(operand.getDefiningOp()))
        continue;

      for (auto *user : operand.getUsers()) {
        if (!loop->isAncestor(user)) {
          isLocal = false;
          break;
        }
      }

      if (isLocal)
        locals.insert(operand);
    }
  });
}

/// For a "loop-local" value \p local within a loop's scope, localizes that
/// value within the scope of the parallel region the loop maps to. Towards that
/// end, this function moves the allocation of \p local within \p allocRegion.
///
/// \param local - the value used exclusively within a loop's scope (see
/// collectLoopLocalValues).
///
/// \param allocRegion - the parallel region where \p local's allocation will be
/// privatized.
///
/// \param rewriter - builder used for updating \p allocRegion.
static void localizeLoopLocalValue(mlir::Value local, mlir::Region &allocRegion,
                                   mlir::ConversionPatternRewriter &rewriter) {
  rewriter.moveOpBefore(local.getDefiningOp(), &allocRegion.front().front());
}
} // namespace looputils

class DoConcurrentConversion
    : public mlir::OpConversionPattern<fir::DoConcurrentOp> {
public:
  using mlir::OpConversionPattern<fir::DoConcurrentOp>::OpConversionPattern;

  DoConcurrentConversion(
      mlir::MLIRContext *context, bool mapToDevice,
      llvm::DenseSet<fir::DoConcurrentOp> &concurrentLoopsToSkip)
      : OpConversionPattern(context), mapToDevice(mapToDevice),
        concurrentLoopsToSkip(concurrentLoopsToSkip) {}

  mlir::LogicalResult
  matchAndRewrite(fir::DoConcurrentOp doLoop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (mapToDevice)
      return doLoop.emitError(
          "not yet implemented: Mapping `do concurrent` loops to device");

    looputils::InductionVariableInfos ivInfos;
    auto loop = mlir::cast<fir::DoConcurrentLoopOp>(
        doLoop.getRegion().back().getTerminator());

    auto indVars = loop.getLoopInductionVars();
    assert(indVars.has_value());

    for (mlir::Value indVar : *indVars)
      ivInfos.emplace_back(loop, indVar);

    llvm::SetVector<mlir::Value> locals;
    looputils::collectLoopLocalValues(loop, locals);

    mlir::IRMapping mapper;
    mlir::omp::ParallelOp parallelOp =
        genParallelOp(doLoop.getLoc(), rewriter, ivInfos, mapper);
    mlir::omp::LoopNestOperands loopNestClauseOps;
    genLoopNestClauseOps(doLoop.getLoc(), rewriter, loop, mapper,
                         loopNestClauseOps);

    for (mlir::Value local : locals)
      looputils::localizeLoopLocalValue(local, parallelOp.getRegion(),
                                        rewriter);

    mlir::omp::LoopNestOp ompLoopNest =
        genWsLoopOp(rewriter, loop, mapper, loopNestClauseOps,
                    /*isComposite=*/mapToDevice);

    rewriter.setInsertionPoint(doLoop);
    fir::FirOpBuilder builder(
        rewriter,
        fir::getKindMapping(doLoop->getParentOfType<mlir::ModuleOp>()));

    // Collect iteration variable(s) allocations so that we can move them
    // outside the `fir.do_concurrent` wrapper (before erasing it).
    llvm::SmallVector<mlir::Operation *> opsToMove;
    for (mlir::Operation &op : llvm::drop_end(doLoop))
      opsToMove.push_back(&op);

    mlir::Block *allocBlock = builder.getAllocaBlock();

    for (mlir::Operation *op : llvm::reverse(opsToMove)) {
      rewriter.moveOpBefore(op, allocBlock, allocBlock->begin());
    }

    // Mark `unordered` loops that are not perfectly nested to be skipped from
    // the legality check of the `ConversionTarget` since we are not interested
    // in mapping them to OpenMP.
    ompLoopNest->walk([&](fir::DoConcurrentOp doLoop) {
      concurrentLoopsToSkip.insert(doLoop);
    });

    rewriter.eraseOp(doLoop);

    return mlir::success();
  }

private:
  mlir::omp::ParallelOp
  genParallelOp(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
                looputils::InductionVariableInfos &ivInfos,
                mlir::IRMapping &mapper) const {
    auto parallelOp = rewriter.create<mlir::omp::ParallelOp>(loc);
    rewriter.createBlock(&parallelOp.getRegion());
    rewriter.setInsertionPoint(rewriter.create<mlir::omp::TerminatorOp>(loc));

    genLoopNestIndVarAllocs(rewriter, ivInfos, mapper);
    return parallelOp;
  }

  void genLoopNestIndVarAllocs(mlir::ConversionPatternRewriter &rewriter,
                               looputils::InductionVariableInfos &ivInfos,
                               mlir::IRMapping &mapper) const {

    for (auto &indVarInfo : ivInfos)
      genInductionVariableAlloc(rewriter, indVarInfo.iterVarMemDef, mapper);
  }

  mlir::Operation *
  genInductionVariableAlloc(mlir::ConversionPatternRewriter &rewriter,
                            mlir::Operation *indVarMemDef,
                            mlir::IRMapping &mapper) const {
    assert(
        indVarMemDef != nullptr &&
        "Induction variable memdef is expected to have a defining operation.");

    llvm::SmallSetVector<mlir::Operation *, 2> indVarDeclareAndAlloc;
    for (auto operand : indVarMemDef->getOperands())
      indVarDeclareAndAlloc.insert(operand.getDefiningOp());
    indVarDeclareAndAlloc.insert(indVarMemDef);

    mlir::Operation *result;
    for (mlir::Operation *opToClone : indVarDeclareAndAlloc)
      result = rewriter.clone(*opToClone, mapper);

    return result;
  }

  void
  genLoopNestClauseOps(mlir::Location loc,
                       mlir::ConversionPatternRewriter &rewriter,
                       fir::DoConcurrentLoopOp loop, mlir::IRMapping &mapper,
                       mlir::omp::LoopNestOperands &loopNestClauseOps) const {
    assert(loopNestClauseOps.loopLowerBounds.empty() &&
           "Loop nest bounds were already emitted!");

    auto populateBounds = [](mlir::Value var,
                             llvm::SmallVectorImpl<mlir::Value> &bounds) {
      bounds.push_back(var.getDefiningOp()->getResult(0));
    };

    for (auto [lb, ub, st] : llvm::zip_equal(
             loop.getLowerBound(), loop.getUpperBound(), loop.getStep())) {
      populateBounds(lb, loopNestClauseOps.loopLowerBounds);
      populateBounds(ub, loopNestClauseOps.loopUpperBounds);
      populateBounds(st, loopNestClauseOps.loopSteps);
    }

    loopNestClauseOps.loopInclusive = rewriter.getUnitAttr();
  }

  mlir::omp::LoopNestOp
  genWsLoopOp(mlir::ConversionPatternRewriter &rewriter,
              fir::DoConcurrentLoopOp loop, mlir::IRMapping &mapper,
              const mlir::omp::LoopNestOperands &clauseOps,
              bool isComposite) const {
    mlir::omp::WsloopOperands wsloopClauseOps;

    // For `local` (and `local_init`) opernads, emit corresponding `private`
    // clauses and attach these clauses to the workshare loop.
    if (!loop.getLocalOperands().empty())
      for (auto [op, sym, arg] : llvm::zip_equal(
               loop.getLocalOperands(),
               loop.getLocalSymsAttr().getAsRange<mlir::SymbolRefAttr>(),
               loop.getRegionLocalArgs())) {
        auto localizer = mlir::SymbolTable::lookupNearestSymbolFrom<
            fir::LocalitySpecifierOp>(loop, sym);
        if (localizer.getLocalitySpecifierType() ==
            fir::LocalitySpecifierType::LocalInit)
          TODO(localizer.getLoc(),
               "local_init conversion is not supported yet");

        auto oldIP = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointAfter(localizer);
        auto privatizer = rewriter.create<mlir::omp::PrivateClauseOp>(
            localizer.getLoc(), sym.getLeafReference().str() + ".omp",
            localizer.getTypeAttr().getValue(),
            mlir::omp::DataSharingClauseType::Private);

        if (!localizer.getInitRegion().empty()) {
          rewriter.cloneRegionBefore(localizer.getInitRegion(),
                                     privatizer.getInitRegion(),
                                     privatizer.getInitRegion().begin());
          auto firYield = mlir::cast<fir::YieldOp>(
              privatizer.getInitRegion().back().getTerminator());
          rewriter.setInsertionPoint(firYield);
          rewriter.create<mlir::omp::YieldOp>(firYield.getLoc(),
                                              firYield.getOperands());
          rewriter.eraseOp(firYield);
        }

        if (!localizer.getDeallocRegion().empty()) {
          rewriter.cloneRegionBefore(localizer.getDeallocRegion(),
                                     privatizer.getDeallocRegion(),
                                     privatizer.getDeallocRegion().begin());
          auto firYield = mlir::cast<fir::YieldOp>(
              privatizer.getDeallocRegion().back().getTerminator());
          rewriter.setInsertionPoint(firYield);
          rewriter.create<mlir::omp::YieldOp>(firYield.getLoc(),
                                              firYield.getOperands());
          rewriter.eraseOp(firYield);
        }

        rewriter.restoreInsertionPoint(oldIP);

        wsloopClauseOps.privateVars.push_back(op);
        wsloopClauseOps.privateSyms.push_back(
            mlir::SymbolRefAttr::get(privatizer));
      }

    auto wsloopOp =
        rewriter.create<mlir::omp::WsloopOp>(loop.getLoc(), wsloopClauseOps);
    wsloopOp.setComposite(isComposite);

    Fortran::common::openmp::EntryBlockArgs wsloopArgs;
    wsloopArgs.priv.vars = wsloopClauseOps.privateVars;
    Fortran::common::openmp::genEntryBlock(rewriter, wsloopArgs,
                                           wsloopOp.getRegion());

    auto loopNestOp =
        rewriter.create<mlir::omp::LoopNestOp>(loop.getLoc(), clauseOps);

    // Clone the loop's body inside the loop nest construct using the
    // mapped values.
    rewriter.cloneRegionBefore(loop.getRegion(), loopNestOp.getRegion(),
                               loopNestOp.getRegion().begin(), mapper);

    rewriter.setInsertionPointToEnd(&loopNestOp.getRegion().back());
    rewriter.create<mlir::omp::YieldOp>(loop->getLoc());

    // `local` region arguments are transferred/cloned from the `do concurrent`
    // loop to the loopnest op when the region is cloned above. Instead, these
    // region arguments should be on the workshare loop's region.
    for (auto [wsloopArg, loopNestArg] :
         llvm::zip_equal(wsloopOp.getRegion().getArguments(),
                         loopNestOp.getRegion().getArguments().drop_front(
                             clauseOps.loopLowerBounds.size())))
      rewriter.replaceAllUsesWith(loopNestArg, wsloopArg);

    for (unsigned i = 0; i < loop.getLocalVars().size(); ++i)
      loopNestOp.getRegion().eraseArgument(clauseOps.loopLowerBounds.size());

    return loopNestOp;
  }

  bool mapToDevice;
  llvm::DenseSet<fir::DoConcurrentOp> &concurrentLoopsToSkip;
};

class DoConcurrentConversionPass
    : public flangomp::impl::DoConcurrentConversionPassBase<
          DoConcurrentConversionPass> {
public:
  DoConcurrentConversionPass() = default;

  DoConcurrentConversionPass(
      const flangomp::DoConcurrentConversionPassOptions &options)
      : DoConcurrentConversionPassBase(options) {}

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    if (func.isDeclaration())
      return;

    mlir::MLIRContext *context = &getContext();

    if (mapTo != flangomp::DoConcurrentMappingKind::DCMK_Host &&
        mapTo != flangomp::DoConcurrentMappingKind::DCMK_Device) {
      mlir::emitWarning(mlir::UnknownLoc::get(context),
                        "DoConcurrentConversionPass: invalid `map-to` value. "
                        "Valid values are: `host` or `device`");
      return;
    }

    llvm::DenseSet<fir::DoConcurrentOp> concurrentLoopsToSkip;
    mlir::RewritePatternSet patterns(context);
    patterns.insert<DoConcurrentConversion>(
        context, mapTo == flangomp::DoConcurrentMappingKind::DCMK_Device,
        concurrentLoopsToSkip);
    mlir::ConversionTarget target(*context);
    target.addDynamicallyLegalOp<fir::DoConcurrentOp>(
        [&](fir::DoConcurrentOp op) {
          return concurrentLoopsToSkip.contains(op);
        });
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });

    if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                               std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
flangomp::createDoConcurrentConversionPass(bool mapToDevice) {
  DoConcurrentConversionPassOptions options;
  options.mapTo = mapToDevice ? flangomp::DoConcurrentMappingKind::DCMK_Device
                              : flangomp::DoConcurrentMappingKind::DCMK_Host;

  return std::make_unique<DoConcurrentConversionPass>(options);
}
