//===- DoConcurrentConversion.cpp -- map `DO CONCURRENT` to OpenMP loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "flang/Optimizer/OpenMP/Utils.h"
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
  InductionVariableInfo(fir::DoLoopOp doLoop) { populateInfo(doLoop); }

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
  ///   fir.do_loop %ind_var = %lb to %ub step %s unordered {
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
  void populateInfo(fir::DoLoopOp doLoop) {
    mlir::Value result = nullptr;

    // Checks if a StoreOp is updating the memref of the loop's iteration
    // variable.
    auto isStoringIV = [&](fir::StoreOp storeOp) {
      // Direct store into the IV memref.
      if (storeOp.getValue() == doLoop.getInductionVar()) {
        indVarUpdateOps.push_back(storeOp);
        return true;
      }

      // Indirect store into the IV memref.
      if (auto convertOp = mlir::dyn_cast<fir::ConvertOp>(
              storeOp.getValue().getDefiningOp())) {
        if (convertOp.getOperand() == doLoop.getInductionVar()) {
          indVarUpdateOps.push_back(convertOp);
          indVarUpdateOps.push_back(storeOp);
          return true;
        }
      }

      return false;
    };

    for (mlir::Operation &op : doLoop) {
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

using LoopNestToIndVarMap =
    llvm::MapVector<fir::DoLoopOp, InductionVariableInfo>;

/// Loop \p innerLoop is considered perfectly-nested inside \p outerLoop iff
/// there are no operations in \p outerloop's body other than:
///
/// 1. the operations needed to assign/update \p outerLoop's induction variable.
/// 2. \p innerLoop itself.
///
/// \p return true if \p innerLoop is perfectly nested inside \p outerLoop
/// according to the above definition.
bool isPerfectlyNested(fir::DoLoopOp outerLoop, fir::DoLoopOp innerLoop) {
  mlir::ForwardSliceOptions forwardSliceOptions;
  forwardSliceOptions.inclusive = true;
  // The following will be used as an example to clarify the internals of this
  // function:
  // ```
  // 1. fir.do_loop %i_idx = %34 to %36 step %c1 unordered {
  // 2.   %i_idx_2 = fir.convert %i_idx : (index) -> i32
  // 3.   fir.store %i_idx_2 to %i_iv#1 : !fir.ref<i32>
  //
  // 4.   fir.do_loop %j_idx = %37 to %39 step %c1_3 unordered {
  // 5.     %j_idx_2 = fir.convert %j_idx : (index) -> i32
  // 6.     fir.store %j_idx_2 to %j_iv#1 : !fir.ref<i32>
  //        ... loop nest body, possible uses %i_idx ...
  //      }
  //    }
  // ```
  // In this example, the `j` loop is perfectly nested inside the `i` loop and
  // below is how we find that.

  // We don't care about the outer-loop's induction variable's uses within the
  // inner-loop, so we filter out these uses.
  //
  // This filter tells `getForwardSlice` (below) to only collect operations
  // which produce results defined above (i.e. outside) the inner-loop's body.
  //
  // Since `outerLoop.getInductionVar()` is a block argument (to the
  // outer-loop's body), the filter effectively collects uses of
  // `outerLoop.getInductionVar()` inside the outer-loop but outside the
  // inner-loop.
  forwardSliceOptions.filter = [&](mlir::Operation *op) {
    return mlir::areValuesDefinedAbove(op->getResults(), innerLoop.getRegion());
  };

  llvm::SetVector<mlir::Operation *> indVarSlice;
  // The forward slice of the `i` loop's IV will be the 2 ops in line 1 & 2
  // above. Uses of `%i_idx` inside the `j` loop are not collected because of
  // the filter.
  mlir::getForwardSlice(outerLoop.getInductionVar(), &indVarSlice,
                        forwardSliceOptions);
  llvm::DenseSet<mlir::Operation *> indVarSet(indVarSlice.begin(),
                                              indVarSlice.end());

  llvm::DenseSet<mlir::Operation *> outerLoopBodySet;
  // The following walk collects ops inside `outerLoop` that are **not**:
  // * the outer-loop itself,
  // * or the inner-loop,
  // * or the `fir.result` op (the outer-loop's terminator).
  //
  // For the above example, this will also populate `outerLoopBodySet` with ops
  // in line 1 & 2 since we skip the `i` loop, the `j` loop, and the terminator.
  outerLoop.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (op == outerLoop)
      return mlir::WalkResult::advance();

    if (op == innerLoop)
      return mlir::WalkResult::skip();

    if (mlir::isa<fir::ResultOp>(op))
      return mlir::WalkResult::advance();

    outerLoopBodySet.insert(op);
    return mlir::WalkResult::advance();
  });

  // If `outerLoopBodySet` ends up having the same ops as `indVarSet`, then
  // `outerLoop` only contains ops that setup its induction variable +
  // `innerLoop` + the `fir.result` terminator. In other words, `innerLoop` is
  // perfectly nested inside `outerLoop`.
  bool result = (outerLoopBodySet == indVarSet);
  LLVM_DEBUG(DBGS() << "Loop pair starting at location " << outerLoop.getLoc()
                    << " is" << (result ? "" : " not")
                    << " perfectly nested\n");

  return result;
}

/// Starting with `currentLoop` collect a perfectly nested loop nest, if any.
/// This function collects as much as possible loops in the nest; it case it
/// fails to recognize a certain nested loop as part of the nest it just returns
/// the parent loops it discovered before.
mlir::LogicalResult collectLoopNest(fir::DoLoopOp currentLoop,
                                    LoopNestToIndVarMap &loopNest) {
  assert(currentLoop.getUnordered());

  while (true) {
    loopNest.insert({currentLoop, InductionVariableInfo(currentLoop)});
    llvm::SmallVector<fir::DoLoopOp> unorderedLoops;

    for (auto nestedLoop : currentLoop.getRegion().getOps<fir::DoLoopOp>())
      if (nestedLoop.getUnordered())
        unorderedLoops.push_back(nestedLoop);

    if (unorderedLoops.empty())
      break;

    // Having more than one unordered loop means that we are not dealing with a
    // perfect loop nest (i.e. a mulit-range `do concurrent` loop); which is the
    // case we are after here.
    if (unorderedLoops.size() > 1)
      return mlir::failure();

    fir::DoLoopOp nestedUnorderedLoop = unorderedLoops.front();

    if (!isPerfectlyNested(currentLoop, nestedUnorderedLoop))
      return mlir::failure();

    currentLoop = nestedUnorderedLoop;
  }

  return mlir::success();
}

/// Prepares the `fir.do_loop` nest to be easily mapped to OpenMP. In
/// particular, this function would take this input IR:
/// ```
/// fir.do_loop %i_iv = %i_lb to %i_ub step %i_step unordered {
///   fir.store %i_iv to %i#1 : !fir.ref<i32>
///   %j_lb = arith.constant 1 : i32
///   %j_ub = arith.constant 10 : i32
///   %j_step = arith.constant 1 : index
///
///   fir.do_loop %j_iv = %j_lb to %j_ub step %j_step unordered {
///     fir.store %j_iv to %j#1 : !fir.ref<i32>
///     ...
///   }
/// }
/// ```
///
/// into the following form (using generic op form since the result is
/// technically an invalid `fir.do_loop` op:
///
/// ```
/// "fir.do_loop"(%i_lb, %i_ub, %i_step) <{unordered}> ({
/// ^bb0(%i_iv: index):
///   %j_lb = "arith.constant"() <{value = 1 : i32}> : () -> i32
///   %j_ub = "arith.constant"() <{value = 10 : i32}> : () -> i32
///   %j_step = "arith.constant"() <{value = 1 : index}> : () -> index
///
///   "fir.do_loop"(%j_lb, %j_ub, %j_step) <{unordered}> ({
///   ^bb0(%new_i_iv: index, %new_j_iv: index):
///     "fir.store"(%new_i_iv, %i#1) : (i32, !fir.ref<i32>) -> ()
///     "fir.store"(%new_j_iv, %j#1) : (i32, !fir.ref<i32>) -> ()
///     ...
///   })
/// ```
///
/// What happened to the loop nest is the following:
///
/// * the innermost loop's entry block was updated from having one operand to
///   having `n` operands where `n` is the number of loops in the nest,
///
/// * the outer loop(s)' ops that update the IVs were sank inside the innermost
///   loop (see the `"fir.store"(%new_i_iv, %i#1)` op above),
///
/// * the innermost loop's entry block's arguments were mapped in order from the
///   outermost to the innermost IV.
///
/// With this IR change, we can directly inline the innermost loop's region into
/// the newly generated `omp.loop_nest` op.
///
/// Note that this function has a pre-condition that \p loopNest consists of
/// perfectly nested loops; i.e. there are no in-between ops between 2 nested
/// loops except for the ops to setup the inner loop's LB, UB, and step. These
/// ops are handled/cloned by `genLoopNestClauseOps(..)`.
void sinkLoopIVArgs(mlir::ConversionPatternRewriter &rewriter,
                    looputils::LoopNestToIndVarMap &loopNest) {
  if (loopNest.size() <= 1)
    return;

  fir::DoLoopOp innermostLoop = loopNest.back().first;
  mlir::Operation &innermostFirstOp = innermostLoop.getRegion().front().front();

  llvm::SmallVector<mlir::Type> argTypes;
  llvm::SmallVector<mlir::Location> argLocs;

  for (auto &[doLoop, indVarInfo] : llvm::drop_end(loopNest)) {
    // Sink the IV update ops to the innermost loop. We need to do for all loops
    // except for the innermost one, hence the `drop_end` usage above.
    for (mlir::Operation *op : indVarInfo.indVarUpdateOps)
      op->moveBefore(&innermostFirstOp);

    argTypes.push_back(doLoop.getInductionVar().getType());
    argLocs.push_back(doLoop.getInductionVar().getLoc());
  }

  mlir::Region &innermmostRegion = innermostLoop.getRegion();
  // Extend the innermost entry block with arguments to represent the outer IVs.
  innermmostRegion.addArguments(argTypes, argLocs);

  unsigned idx = 1;
  // In reverse, remap the IVs of the loop nest from the old values to the new
  // ones. We do that in reverse since the first argument before this loop is
  // the old IV for the innermost loop. Therefore, we want to replace it first
  // before the old value (1st argument in the block) is remapped to be the IV
  // of the outermost loop in the nest.
  for (auto &[doLoop, _] : llvm::reverse(loopNest)) {
    doLoop.getInductionVar().replaceAllUsesWith(
        innermmostRegion.getArgument(innermmostRegion.getNumArguments() - idx));
    ++idx;
  }
}

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
void collectLoopLocalValues(fir::DoLoopOp doLoop,
                            llvm::SetVector<mlir::Value> &locals) {
  doLoop.walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands()) {
      if (locals.contains(operand))
        continue;

      bool isLocal = true;

      if (!mlir::isa_and_present<fir::AllocaOp>(operand.getDefiningOp()))
        continue;

      // Values defined inside the loop are not interesting since they do not
      // need to be localized.
      if (doLoop->isAncestor(operand.getDefiningOp()))
        continue;

      for (auto *user : operand.getUsers()) {
        if (!doLoop->isAncestor(user)) {
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

class DoConcurrentConversion : public mlir::OpConversionPattern<fir::DoLoopOp> {
public:
  using mlir::OpConversionPattern<fir::DoLoopOp>::OpConversionPattern;

  DoConcurrentConversion(mlir::MLIRContext *context, bool mapToDevice,
                         llvm::DenseSet<fir::DoLoopOp> &concurrentLoopsToSkip)
      : OpConversionPattern(context), mapToDevice(mapToDevice),
        concurrentLoopsToSkip(concurrentLoopsToSkip) {}

  mlir::LogicalResult
  matchAndRewrite(fir::DoLoopOp doLoop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (mapToDevice)
      return doLoop.emitError(
          "not yet implemented: Mapping `do concurrent` loops to device");

    looputils::LoopNestToIndVarMap loopNest;
    bool hasRemainingNestedLoops =
        failed(looputils::collectLoopNest(doLoop, loopNest));
    if (hasRemainingNestedLoops)
      mlir::emitWarning(doLoop.getLoc(),
                        "Some `do concurent` loops are not perfectly-nested. "
                        "These will be serialized.");

    llvm::SetVector<mlir::Value> locals;
    looputils::collectLoopLocalValues(loopNest.back().first, locals);
    looputils::sinkLoopIVArgs(rewriter, loopNest);

    mlir::IRMapping mapper;
    mlir::omp::ParallelOp parallelOp =
        genParallelOp(doLoop.getLoc(), rewriter, loopNest, mapper);
    mlir::omp::LoopNestOperands loopNestClauseOps;
    genLoopNestClauseOps(doLoop.getLoc(), rewriter, loopNest, mapper,
                         loopNestClauseOps);

    for (mlir::Value local : locals)
      looputils::localizeLoopLocalValue(local, parallelOp.getRegion(),
                                        rewriter);

    mlir::omp::LoopNestOp ompLoopNest =
        genWsLoopOp(rewriter, loopNest.back().first, mapper, loopNestClauseOps,
                    /*isComposite=*/mapToDevice);

    rewriter.eraseOp(doLoop);

    // Mark `unordered` loops that are not perfectly nested to be skipped from
    // the legality check of the `ConversionTarget` since we are not interested
    // in mapping them to OpenMP.
    ompLoopNest->walk([&](fir::DoLoopOp doLoop) {
      if (doLoop.getUnordered()) {
        concurrentLoopsToSkip.insert(doLoop);
      }
    });

    return mlir::success();
  }

private:
  mlir::omp::ParallelOp genParallelOp(mlir::Location loc,
                                      mlir::ConversionPatternRewriter &rewriter,
                                      looputils::LoopNestToIndVarMap &loopNest,
                                      mlir::IRMapping &mapper) const {
    auto parallelOp = rewriter.create<mlir::omp::ParallelOp>(loc);
    rewriter.createBlock(&parallelOp.getRegion());
    rewriter.setInsertionPoint(rewriter.create<mlir::omp::TerminatorOp>(loc));

    genLoopNestIndVarAllocs(rewriter, loopNest, mapper);
    return parallelOp;
  }

  void genLoopNestIndVarAllocs(mlir::ConversionPatternRewriter &rewriter,
                               looputils::LoopNestToIndVarMap &loopNest,
                               mlir::IRMapping &mapper) const {

    for (auto &[_, indVarInfo] : loopNest)
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

  void genLoopNestClauseOps(
      mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
      looputils::LoopNestToIndVarMap &loopNest, mlir::IRMapping &mapper,
      mlir::omp::LoopNestOperands &loopNestClauseOps) const {
    assert(loopNestClauseOps.loopLowerBounds.empty() &&
           "Loop nest bounds were already emitted!");

    auto populateBounds = [](mlir::Value var,
                             llvm::SmallVectorImpl<mlir::Value> &bounds) {
      bounds.push_back(var.getDefiningOp()->getResult(0));
    };

    for (auto &[doLoop, _] : loopNest) {
      populateBounds(doLoop.getLowerBound(), loopNestClauseOps.loopLowerBounds);
      populateBounds(doLoop.getUpperBound(), loopNestClauseOps.loopUpperBounds);
      populateBounds(doLoop.getStep(), loopNestClauseOps.loopSteps);
    }

    loopNestClauseOps.loopInclusive = rewriter.getUnitAttr();
  }

  mlir::omp::LoopNestOp
  genWsLoopOp(mlir::ConversionPatternRewriter &rewriter, fir::DoLoopOp doLoop,
              mlir::IRMapping &mapper,
              const mlir::omp::LoopNestOperands &clauseOps,
              bool isComposite) const {

    auto wsloopOp = rewriter.create<mlir::omp::WsloopOp>(doLoop.getLoc());
    wsloopOp.setComposite(isComposite);
    rewriter.createBlock(&wsloopOp.getRegion());

    auto loopNestOp =
        rewriter.create<mlir::omp::LoopNestOp>(doLoop.getLoc(), clauseOps);

    // Clone the loop's body inside the loop nest construct using the
    // mapped values.
    rewriter.cloneRegionBefore(doLoop.getRegion(), loopNestOp.getRegion(),
                               loopNestOp.getRegion().begin(), mapper);

    mlir::Operation *terminator = loopNestOp.getRegion().back().getTerminator();
    rewriter.setInsertionPointToEnd(&loopNestOp.getRegion().back());
    rewriter.create<mlir::omp::YieldOp>(terminator->getLoc());
    rewriter.eraseOp(terminator);

    return loopNestOp;
  }

  bool mapToDevice;
  llvm::DenseSet<fir::DoLoopOp> &concurrentLoopsToSkip;
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

    llvm::DenseSet<fir::DoLoopOp> concurrentLoopsToSkip;
    mlir::RewritePatternSet patterns(context);
    patterns.insert<DoConcurrentConversion>(
        context, mapTo == flangomp::DoConcurrentMappingKind::DCMK_Device,
        concurrentLoopsToSkip);
    mlir::ConversionTarget target(*context);
    target.addDynamicallyLegalOp<fir::DoLoopOp>([&](fir::DoLoopOp op) {
      // The goal is to handle constructs that eventually get lowered to
      // `fir.do_loop` with the `unordered` attribute (e.g. array expressions).
      // Currently, this is only enabled for the `do concurrent` construct since
      // the pass runs early in the pipeline.
      return !op.getUnordered() || concurrentLoopsToSkip.contains(op);
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
