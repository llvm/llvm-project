//===- DoConcurrentConversion.cpp -- map `DO CONCURRENT` to OpenMP loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace flangomp {
#define GEN_PASS_DEF_DOCONCURRENTCONVERSIONPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "do-concurrent-conversion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace Fortran {
namespace lower {
namespace omp {
namespace internal {
// TODO The following 2 functions are copied from "flang/Lower/OpenMP/Utils.h".
// This duplication is temporary until we find a solution for a shared location
// for these utils that does not introduce circular CMake deps.
mlir::omp::MapInfoOp createMapInfoOp(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value baseAddr,
    mlir::Value varPtrPtr, std::string name, llvm::ArrayRef<mlir::Value> bounds,
    llvm::ArrayRef<mlir::Value> members, mlir::ArrayAttr membersIndex,
    uint64_t mapType, mlir::omp::VariableCaptureKind mapCaptureType,
    mlir::Type retTy, bool partialMap = false,
    mlir::FlatSymbolRefAttr mapperId = mlir::FlatSymbolRefAttr()) {
  if (auto boxTy = llvm::dyn_cast<fir::BaseBoxType>(baseAddr.getType())) {
    baseAddr = builder.create<fir::BoxAddrOp>(loc, baseAddr);
    retTy = baseAddr.getType();
  }

  mlir::TypeAttr varType = mlir::TypeAttr::get(
      llvm::cast<mlir::omp::PointerLikeType>(retTy).getElementType());

  // For types with unknown extents such as <2x?xi32> we discard the incomplete
  // type info and only retain the base type. The correct dimensions are later
  // recovered through the bounds info.
  if (auto seqType = llvm::dyn_cast<fir::SequenceType>(varType.getValue()))
    if (seqType.hasDynamicExtents())
      varType = mlir::TypeAttr::get(seqType.getEleTy());

  mlir::omp::MapInfoOp op = builder.create<mlir::omp::MapInfoOp>(
      loc, retTy, baseAddr, varType, varPtrPtr, members, membersIndex, bounds,
      builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
      mapperId,
      builder.getAttr<mlir::omp::VariableCaptureKindAttr>(mapCaptureType),
      builder.getStringAttr(name), builder.getBoolAttr(partialMap));

  return op;
}

mlir::Value mapTemporaryValue(fir::FirOpBuilder &builder,
                              mlir::omp::TargetOp targetOp, mlir::Value val,
                              llvm::StringRef name) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(val);
  auto copyVal = builder.createTemporary(val.getLoc(), val.getType());
  builder.createStoreWithConvert(copyVal.getLoc(), val, copyVal);

  llvm::SmallVector<mlir::Value> bounds;
  builder.setInsertionPoint(targetOp);
  mlir::Value mapOp = createMapInfoOp(
      builder, copyVal.getLoc(), copyVal,
      /*varPtrPtr=*/mlir::Value{}, name.str(), bounds,
      /*members=*/llvm::SmallVector<mlir::Value>{},
      /*membersIndex=*/mlir::ArrayAttr{},
      static_cast<std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT),
      mlir::omp::VariableCaptureKind::ByCopy, copyVal.getType());

  mlir::Region &targetRegion = targetOp.getRegion();

  auto argIface = llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*targetOp);
  unsigned insertIndex =
      argIface.getMapBlockArgsStart() + argIface.numMapBlockArgs();
  targetOp.getMapVarsMutable().append(mlir::ValueRange{mapOp});
  mlir::Value clonedValArg =
      targetRegion.insertArgument(insertIndex, mapOp.getType(), mapOp.getLoc());

  mlir::Block *targetEntryBlock = &targetRegion.getBlocks().front();
  builder.setInsertionPointToStart(targetEntryBlock);
  auto loadOp =
      builder.create<fir::LoadOp>(clonedValArg.getLoc(), clonedValArg);
  return loadOp.getResult();
}

/// Check if cloning the bounds introduced any dependency on the outer region.
/// If so, then either clone them as well if they are MemoryEffectFree, or else
/// copy them to a new temporary and add them to the map and block_argument
/// lists and replace their uses with the new temporary.
///
/// TODO: similar to the above functions, this is copied from OpenMP lowering
/// (in this case, from `genBodyOfTargetOp`). Once we move to a common lib for
/// these utils this will move as well.
void cloneOrMapRegionOutsiders(fir::FirOpBuilder &builder,
                               mlir::omp::TargetOp targetOp) {
  mlir::Region &targetRegion = targetOp.getRegion();
  mlir::Block *targetEntryBlock = &targetRegion.getBlocks().front();
  llvm::SetVector<mlir::Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(targetRegion, valuesDefinedAbove);

  while (!valuesDefinedAbove.empty()) {
    for (mlir::Value val : valuesDefinedAbove) {
      mlir::Operation *valOp = val.getDefiningOp();
      assert(valOp != nullptr);
      if (mlir::isMemoryEffectFree(valOp)) {
        mlir::Operation *clonedOp = valOp->clone();
        targetEntryBlock->push_front(clonedOp);
        assert(clonedOp->getNumResults() == 1);
        val.replaceUsesWithIf(
            clonedOp->getResult(0), [targetEntryBlock](mlir::OpOperand &use) {
              return use.getOwner()->getBlock() == targetEntryBlock;
            });
      } else {
        mlir::Value mappedTemp = mapTemporaryValue(builder, targetOp, val,
                                                   /*name=*/llvm::StringRef{});
        val.replaceUsesWithIf(
            mappedTemp, [targetEntryBlock](mlir::OpOperand &use) {
              return use.getOwner()->getBlock() == targetEntryBlock;
            });
      }
    }
    valuesDefinedAbove.clear();
    mlir::getUsedValuesDefinedAbove(targetRegion, valuesDefinedAbove);
  }
}
} // namespace internal
} // namespace omp
} // namespace lower
} // namespace Fortran

namespace {
namespace looputils {
/// Stores info needed about the induction/iteration variable for each `do
/// concurrent` in a loop nest. This includes:
/// * the operation allocating memory for iteration variable,
/// * the operation(s) updating the iteration variable with the current
///   iteration number.
struct InductionVariableInfo {
  mlir::Operation *iterVarMemDef;
  llvm::SetVector<mlir::Operation *> indVarUpdateOps;
};

using LoopNestToIndVarMap =
    llvm::MapVector<fir::DoLoopOp, InductionVariableInfo>;

/// Given an operation `op`, this returns true if `op`'s operand is ultimately
/// the loop's induction variable. Detecting this helps finding the live-in
/// value corresponding to the induction variable in case the induction variable
/// is indirectly used in the loop (e.g. throught a cast op).
bool isIndVarUltimateOperand(mlir::Operation *op, fir::DoLoopOp doLoop) {
  while (op != nullptr && op->getNumOperands() > 0) {
    auto ivIt = llvm::find_if(op->getOperands(), [&](mlir::Value operand) {
      return operand == doLoop.getInductionVar();
    });

    if (ivIt != op->getOperands().end())
      return true;

    op = op->getOperand(0).getDefiningOp();
  }

  return false;
}

/// For the \p doLoop parameter, find the operations that declares its induction
/// variable or allocates memory for it.
mlir::Operation *findLoopIndVarMemDecl(fir::DoLoopOp doLoop) {
  mlir::Value result = nullptr;
  mlir::visitUsedValuesDefinedAbove(
      doLoop.getRegion(), [&](mlir::OpOperand *operand) {
        if (isIndVarUltimateOperand(operand->getOwner(), doLoop)) {
          assert(result == nullptr &&
                 "loop can have only one induction variable");
          result = operand->get();
        }
      });

  assert(result != nullptr && result.getDefiningOp() != nullptr);
  return result.getDefiningOp();
}

/// Collect the list of values used inside the loop but defined outside of it.
void collectLoopLiveIns(fir::DoLoopOp doLoop,
                        llvm::SmallVectorImpl<mlir::Value> &liveIns) {
  llvm::SmallDenseSet<mlir::Value> seenValues;
  llvm::SmallDenseSet<mlir::Operation *> seenOps;

  mlir::visitUsedValuesDefinedAbove(
      doLoop.getRegion(), [&](mlir::OpOperand *operand) {
        if (!seenValues.insert(operand->get()).second)
          return;

        mlir::Operation *definingOp = operand->get().getDefiningOp();
        // We want to collect ops corresponding to live-ins only once.
        if (definingOp && !seenOps.insert(definingOp).second)
          return;

        liveIns.push_back(operand->get());
      });
}

/// Collects the op(s) responsible for updating a loop's iteration variable with
/// the current iteration number. For example, for the input IR:
/// ```
/// %i = fir.alloca i32 {bindc_name = "i"}
/// %i_decl:2 = hlfir.declare %i ...
/// ...
/// fir.do_loop %i_iv = %lb to %ub step %step unordered {
///   %1 = fir.convert %i_iv : (index) -> i32
///   fir.store %1 to %i_decl#1 : !fir.ref<i32>
///   ...
/// }
/// ```
/// this function would return the first 2 ops in the `fir.do_loop`'s region.
llvm::SetVector<mlir::Operation *>
extractIndVarUpdateOps(fir::DoLoopOp doLoop) {
  mlir::Value indVar = doLoop.getInductionVar();
  llvm::SetVector<mlir::Operation *> indVarUpdateOps;

  llvm::SmallVector<mlir::Value> toProcess;
  toProcess.push_back(indVar);

  llvm::DenseSet<mlir::Value> done;

  while (!toProcess.empty()) {
    mlir::Value val = toProcess.back();
    toProcess.pop_back();

    if (!done.insert(val).second)
      continue;

    for (mlir::Operation *user : val.getUsers()) {
      indVarUpdateOps.insert(user);

      for (mlir::Value result : user->getResults())
        toProcess.push_back(result);
    }
  }

  return std::move(indVarUpdateOps);
}

/// Starting with a value at the end of a definition/conversion chain, walk the
/// chain backwards and collect all the visited ops along the way. This is the
/// same as the "backward slice" of the use-def chain of \p link.
///
/// If the root of the chain/slice is a constant op  (where convert operations
/// on constant count as constants as well), then populate \p opChain with the
/// extracted chain/slice. If not, then \p opChain will contains a single value:
/// \p link.
///
/// The purpose of this function is that we pull in the chain of
/// constant+conversion ops inside the parallel region if possible; which
/// prevents creating an unnecessary shared/mapped value that crosses the OpenMP
/// region.
///
/// For example, given this IR:
/// ```
/// %c10 = arith.constant 10 : i32
/// %10 = fir.convert %c10 : (i32) -> index
/// ```
/// and giving `%10` as the starting input: `link`, `defChain` would contain
/// both of the above ops.
void collectIndirectConstOpChain(mlir::Operation *link,
                                 llvm::SetVector<mlir::Operation *> &opChain) {
  mlir::BackwardSliceOptions options;
  options.inclusive = true;
  mlir::getBackwardSlice(link, &opChain, options);

  assert(!opChain.empty());

  bool isConstantChain = [&]() {
    if (!mlir::isa_and_present<mlir::arith::ConstantOp>(opChain.front()))
      return false;

    return llvm::all_of(llvm::drop_begin(opChain), [](mlir::Operation *op) {
      return mlir::isa_and_present<fir::ConvertOp>(op);
    });
  }();

  if (isConstantChain)
    return;

  opChain.clear();
  opChain.insert(link);
}

/// Loop \p innerLoop is considered perfectly-nested inside \p outerLoop iff
/// there are no operations in \p outerloop's other than:
///
/// 1. the operations needed to assing/update \p outerLoop's induction variable.
/// 2. \p innerLoop itself.
///
/// \p return true if \p innerLoop is perfectly nested inside \p outerLoop
/// according to the above definition.
bool isPerfectlyNested(fir::DoLoopOp outerLoop, fir::DoLoopOp innerLoop) {
  mlir::BackwardSliceOptions backwardSliceOptions;
  backwardSliceOptions.inclusive = true;
  // We will collect the backward slices for innerLoop's LB, UB, and step.
  // However, we want to limit the scope of these slices to the scope of
  // outerLoop's region.
  backwardSliceOptions.filter = [&](mlir::Operation *op) {
    return !mlir::areValuesDefinedAbove(op->getResults(),
                                        outerLoop.getRegion());
  };

  mlir::ForwardSliceOptions forwardSliceOptions;
  forwardSliceOptions.inclusive = true;
  // We don't care about the outer-loop's induction variable's uses within the
  // inner-loop, so we filter out these uses.
  forwardSliceOptions.filter = [&](mlir::Operation *op) {
    return mlir::areValuesDefinedAbove(op->getResults(), innerLoop.getRegion());
  };

  llvm::SetVector<mlir::Operation *> indVarSlice;
  mlir::getForwardSlice(outerLoop.getInductionVar(), &indVarSlice,
                        forwardSliceOptions);
  llvm::DenseSet<mlir::Operation *> innerLoopSetupOpsSet(indVarSlice.begin(),
                                                         indVarSlice.end());

  llvm::DenseSet<mlir::Operation *> loopBodySet;
  outerLoop.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (op == outerLoop)
      return mlir::WalkResult::advance();

    if (op == innerLoop)
      return mlir::WalkResult::skip();

    if (mlir::isa<fir::ResultOp>(op))
      return mlir::WalkResult::advance();

    loopBodySet.insert(op);
    return mlir::WalkResult::advance();
  });

  bool result = (loopBodySet == innerLoopSetupOpsSet);
  mlir::Location loc = outerLoop.getLoc();
  LLVM_DEBUG(DBGS() << "Loop pair starting at location " << loc << " is"
                    << (result ? "" : " not") << " perfectly nested\n");

  return result;
}

/// Starting with `outerLoop` collect a perfectly nested loop nest, if any. This
/// function collects as much as possible loops in the nest; it case it fails to
/// recognize a certain nested loop as part of the nest it just returns the
/// parent loops it discovered before.
mlir::LogicalResult collectLoopNest(fir::DoLoopOp currentLoop,
                                    LoopNestToIndVarMap &loopNest) {
  assert(currentLoop.getUnordered());

  while (true) {
    loopNest.try_emplace(
        currentLoop,
        InductionVariableInfo{
            findLoopIndVarMemDecl(currentLoop),
            std::move(looputils::extractIndVarUpdateOps(currentLoop))});

    auto directlyNestedLoops = currentLoop.getRegion().getOps<fir::DoLoopOp>();
    llvm::SmallVector<fir::DoLoopOp> unorderedLoops;

    for (auto nestedLoop : directlyNestedLoops)
      if (nestedLoop.getUnordered())
        unorderedLoops.push_back(nestedLoop);

    if (unorderedLoops.empty())
      break;

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
/// \param [in] doLoop - the loop within which the function searches for values
/// used exclusively inside.
///
/// \param [out] locals - the list of loop-local values detected for \p doLoop.
static void collectLoopLocalValues(fir::DoLoopOp doLoop,
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
    mlir::Operation *lbOp = doLoop.getLowerBound().getDefiningOp();
    mlir::Operation *ubOp = doLoop.getUpperBound().getDefiningOp();
    mlir::Operation *stepOp = doLoop.getStep().getDefiningOp();

    if (lbOp == nullptr || ubOp == nullptr || stepOp == nullptr) {
      return rewriter.notifyMatchFailure(
          doLoop, "At least one of the loop's LB, UB, or step doesn't have a "
                  "defining operation.");
    }

    looputils::LoopNestToIndVarMap loopNest;
    bool hasRemainingNestedLoops =
        failed(looputils::collectLoopNest(doLoop, loopNest));
    if (hasRemainingNestedLoops)
      mlir::emitWarning(doLoop.getLoc(),
                        "Some `do concurent` loops are not perfectly-nested. "
                        "These will be serialzied.");

    llvm::SmallVector<mlir::Value> loopNestLiveIns;
    looputils::collectLoopLiveIns(loopNest.back().first, loopNestLiveIns);
    assert(!loopNestLiveIns.empty());

    llvm::SetVector<mlir::Value> locals;
    looputils::collectLoopLocalValues(loopNest.back().first, locals);
    // We do not want to map "loop-local" values to the device through
    // `omp.map.info` ops. Therefore, we remove them from the list of live-ins.
    loopNestLiveIns.erase(llvm::remove_if(loopNestLiveIns,
                                          [&](mlir::Value liveIn) {
                                            return locals.contains(liveIn);
                                          }),
                          loopNestLiveIns.end());

    looputils::sinkLoopIVArgs(rewriter, loopNest);

    mlir::omp::TargetOp targetOp;
    mlir::omp::LoopNestOperands loopNestClauseOps;

    mlir::IRMapping mapper;

    if (mapToDevice) {
      // TODO: Currently the loop bounds for the outer loop are duplicated.
      mlir::omp::TargetOperands targetClauseOps;
      genLoopNestClauseOps(doLoop.getLoc(), rewriter, loopNest, mapper,
                           loopNestClauseOps, &targetClauseOps);

      // Prevent mapping host-evaluated variables.
      loopNestLiveIns.erase(llvm::remove_if(loopNestLiveIns,
                                            [&](mlir::Value liveIn) {
                                              return llvm::is_contained(
                                                  targetClauseOps.hostEvalVars,
                                                  liveIn);
                                            }),
                            loopNestLiveIns.end());

      LiveInShapeInfoMap liveInShapeInfoMap;
      // The outermost loop will contain all the live-in values in all nested
      // loops since live-in values are collected recursively for all nested
      // ops.
      for (mlir::Value liveIn : loopNestLiveIns)
        targetClauseOps.mapVars.push_back(genMapInfoOpForLiveIn(
            rewriter, liveIn, liveInShapeInfoMap[liveIn]));

      targetOp =
          genTargetOp(doLoop.getLoc(), rewriter, mapper, loopNestLiveIns,
                      targetClauseOps, loopNestClauseOps, liveInShapeInfoMap);
      genTeamsOp(doLoop.getLoc(), rewriter);
    }

    mlir::omp::ParallelOp parallelOp =
        genParallelOp(doLoop.getLoc(), rewriter, loopNest, mapper);
    // Only set as composite when part of `distribute parallel do`.
    parallelOp.setComposite(mapToDevice);

    if (!mapToDevice)
      genLoopNestClauseOps(doLoop.getLoc(), rewriter, loopNest, mapper,
                           loopNestClauseOps);

    for (mlir::Value local : locals)
      looputils::localizeLoopLocalValue(local, parallelOp.getRegion(),
                                        rewriter);

    if (mapToDevice)
      genDistributeOp(doLoop.getLoc(), rewriter).setComposite(/*val=*/true);

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
  struct TargetDeclareShapeCreationInfo {
    // Note: We use `std::vector` (rather than `llvm::SmallVector` as usual) to
    // interface more easily `ShapeShiftOp::getOrigins()` which returns
    // `std::vector`.
    std::vector<mlir::Value> startIndices{};
    std::vector<mlir::Value> extents{};

    bool isShapedValue() const { return !extents.empty(); }
    bool isShapeShiftedValue() const { return !startIndices.empty(); }
  };

  using LiveInShapeInfoMap =
      llvm::DenseMap<mlir::Value, TargetDeclareShapeCreationInfo>;

  void
  genBoundsOps(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
               mlir::Value shape, llvm::SmallVectorImpl<mlir::Value> &boundsOps,
               TargetDeclareShapeCreationInfo &targetShapeCreationInfo) const {
    if (shape == nullptr) {
      return;
    }

    auto shapeOp =
        mlir::dyn_cast_if_present<fir::ShapeOp>(shape.getDefiningOp());
    auto shapeShiftOp =
        mlir::dyn_cast_if_present<fir::ShapeShiftOp>(shape.getDefiningOp());

    if (shapeOp == nullptr && shapeShiftOp == nullptr)
      TODO(loc,
           "Shapes not defined by `fir.shape` or `fir.shape_shift` op's are "
           "not supported yet.");

    auto extents = shapeOp != nullptr
                       ? std::vector<mlir::Value>(shapeOp.getExtents().begin(),
                                                  shapeOp.getExtents().end())
                       : shapeShiftOp.getExtents();

    mlir::Type idxType = extents.front().getType();

    auto one = rewriter.create<mlir::arith::ConstantOp>(
        loc, idxType, rewriter.getIntegerAttr(idxType, 1));
    // For non-shifted values, that starting index is the default Fortran
    // value: 1.
    std::vector<mlir::Value> startIndices =
        shapeOp != nullptr ? std::vector<mlir::Value>(extents.size(), one)
                           : shapeShiftOp.getOrigins();

    auto genBoundsOp = [&](mlir::Value startIndex, mlir::Value extent) {
      // We map the entire range of data by default, therefore, we always map
      // from the start.
      auto normalizedLB = rewriter.create<mlir::arith::ConstantOp>(
          loc, idxType, rewriter.getIntegerAttr(idxType, 0));

      mlir::Value ub = rewriter.create<mlir::arith::SubIOp>(loc, extent, one);

      return rewriter.create<mlir::omp::MapBoundsOp>(
          loc, rewriter.getType<mlir::omp::MapBoundsType>(), normalizedLB, ub,
          extent,
          /*stride=*/mlir::Value{}, /*stride_in_bytes=*/false, startIndex);
    };

    for (auto [startIndex, extent] : llvm::zip_equal(startIndices, extents))
      boundsOps.push_back(genBoundsOp(startIndex, extent));

    if (shapeShiftOp != nullptr)
      targetShapeCreationInfo.startIndices = std::move(startIndices);
    targetShapeCreationInfo.extents = std::move(extents);
  }

  mlir::omp::MapInfoOp genMapInfoOpForLiveIn(
      mlir::ConversionPatternRewriter &rewriter, mlir::Value liveIn,
      TargetDeclareShapeCreationInfo &targetShapeCreationInfo) const {
    mlir::Value rawAddr = liveIn;
    mlir::Value shape = nullptr;
    llvm::StringRef name;

    mlir::Operation *liveInDefiningOp = liveIn.getDefiningOp();
    auto declareOp =
        mlir::dyn_cast_if_present<hlfir::DeclareOp>(liveInDefiningOp);

    if (declareOp != nullptr) {
      // Use the raw address to avoid unboxing `fir.box` values whenever
      // possible. Put differently, if we have access to the direct value memory
      // reference/address, we use it.
      rawAddr = declareOp.getOriginalBase();
      shape = declareOp.getShape();
      name = declareOp.getUniqName();
    }

    if (!llvm::isa<mlir::omp::PointerLikeType>(rawAddr.getType())) {
      fir::FirOpBuilder builder(
          rewriter, fir::getKindMapping(
                        liveInDefiningOp->getParentOfType<mlir::ModuleOp>()));
      builder.setInsertionPointAfter(liveInDefiningOp);
      auto copyVal = builder.createTemporary(liveIn.getLoc(), liveIn.getType());
      builder.createStoreWithConvert(copyVal.getLoc(), liveIn, copyVal);
      rawAddr = copyVal;
    }

    mlir::Type liveInType = liveIn.getType();
    mlir::Type eleType = liveInType;
    if (auto refType = mlir::dyn_cast<fir::ReferenceType>(liveInType))
      eleType = refType.getElementType();

    llvm::omp::OpenMPOffloadMappingFlags mapFlag =
        llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;
    mlir::omp::VariableCaptureKind captureKind =
        mlir::omp::VariableCaptureKind::ByRef;

    if (fir::isa_trivial(eleType) || fir::isa_char(eleType)) {
      captureKind = mlir::omp::VariableCaptureKind::ByCopy;
    } else if (!fir::isa_builtin_cptr_type(eleType)) {
      mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
      mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
    }

    llvm::SmallVector<mlir::Value> boundsOps;
    genBoundsOps(rewriter, liveIn.getLoc(), shape, boundsOps,
                 targetShapeCreationInfo);

    return Fortran::lower::omp::internal::createMapInfoOp(
        rewriter, liveIn.getLoc(), rawAddr,
        /*varPtrPtr=*/{}, name.str(), boundsOps,
        /*members=*/{},
        /*membersIndex=*/mlir::ArrayAttr{},
        static_cast<
            std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
            mapFlag),
        captureKind, rawAddr.getType());
  }

  mlir::omp::TargetOp
  genTargetOp(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
              mlir::IRMapping &mapper, llvm::ArrayRef<mlir::Value> mappedVars,
              mlir::omp::TargetOperands &clauseOps,
              mlir::omp::LoopNestOperands &loopNestClauseOps,
              const LiveInShapeInfoMap &liveInShapeInfoMap) const {
    auto targetOp = rewriter.create<mlir::omp::TargetOp>(loc, clauseOps);
    auto argIface = llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*targetOp);

    mlir::Region &region = targetOp.getRegion();

    llvm::SmallVector<mlir::Type> regionArgTypes;
    llvm::SmallVector<mlir::Location> regionArgLocs;

    for (auto var : llvm::concat<const mlir::Value>(clauseOps.hostEvalVars,
                                                    clauseOps.mapVars)) {
      regionArgTypes.push_back(var.getType());
      regionArgLocs.push_back(var.getLoc());
    }

    rewriter.createBlock(&region, {}, regionArgTypes, regionArgLocs);
    fir::FirOpBuilder builder(
        rewriter,
        fir::getKindMapping(targetOp->getParentOfType<mlir::ModuleOp>()));

    // Within the loop, it possible that we discover other values that need to
    // mapped to the target region (the shape info values for arrays, for
    // example). Therefore, the map block args might be extended and resized.
    // Hence, we invoke `argIface.getMapBlockArgs()` every iteration to make
    // sure we access the proper vector of data.
    int idx = 0;
    for (auto [mapInfoOp, mappedVar] :
         llvm::zip_equal(clauseOps.mapVars, mappedVars)) {
      auto miOp = mlir::cast<mlir::omp::MapInfoOp>(mapInfoOp.getDefiningOp());
      hlfir::DeclareOp liveInDeclare =
          genLiveInDeclare(builder, targetOp, argIface.getMapBlockArgs()[idx],
                           miOp, liveInShapeInfoMap.at(mappedVar));
      ++idx;

      // TODO If `mappedVar.getDefiningOp()` is a `fir::BoxAddrOp`, we probably
      // need to "unpack" the box by getting the defining op of it's value.
      // However, we did not hit this case in reality yet so leaving it as a
      // todo for now.

      auto mapHostValueToDevice = [&](mlir::Value hostValue,
                                      mlir::Value deviceValue) {
        if (!llvm::isa<mlir::omp::PointerLikeType>(hostValue.getType()))
          mapper.map(hostValue,
                     builder.loadIfRef(hostValue.getLoc(), deviceValue));
        else
          mapper.map(hostValue, deviceValue);
      };

      mapHostValueToDevice(mappedVar, liveInDeclare.getOriginalBase());

      if (auto origDeclareOp = mlir::dyn_cast_if_present<hlfir::DeclareOp>(
              mappedVar.getDefiningOp()))
        mapHostValueToDevice(origDeclareOp.getBase(), liveInDeclare.getBase());
    }

    for (auto [arg, hostEval] : llvm::zip_equal(argIface.getHostEvalBlockArgs(),
                                                clauseOps.hostEvalVars))
      mapper.map(hostEval, arg);

    for (unsigned i = 0; i < loopNestClauseOps.loopLowerBounds.size(); ++i) {
      loopNestClauseOps.loopLowerBounds[i] =
          mapper.lookup(loopNestClauseOps.loopLowerBounds[i]);
      loopNestClauseOps.loopUpperBounds[i] =
          mapper.lookup(loopNestClauseOps.loopUpperBounds[i]);
      loopNestClauseOps.loopSteps[i] =
          mapper.lookup(loopNestClauseOps.loopSteps[i]);
    }

    Fortran::lower::omp::internal::cloneOrMapRegionOutsiders(builder, targetOp);
    rewriter.setInsertionPoint(
        rewriter.create<mlir::omp::TerminatorOp>(targetOp.getLoc()));

    return targetOp;
  }

  hlfir::DeclareOp genLiveInDeclare(
      fir::FirOpBuilder &builder, mlir::omp::TargetOp targetOp,
      mlir::Value liveInArg, mlir::omp::MapInfoOp liveInMapInfoOp,
      const TargetDeclareShapeCreationInfo &targetShapeCreationInfo) const {
    mlir::Type liveInType = liveInArg.getType();
    std::string liveInName = liveInMapInfoOp.getName().has_value()
                                 ? liveInMapInfoOp.getName().value().str()
                                 : std::string("");
    if (fir::isa_ref_type(liveInType))
      liveInType = fir::unwrapRefType(liveInType);

    mlir::Value shape = [&]() -> mlir::Value {
      if (!targetShapeCreationInfo.isShapedValue())
        return {};

      llvm::SmallVector<mlir::Value> extentOperands;
      llvm::SmallVector<mlir::Value> startIndexOperands;

      if (targetShapeCreationInfo.isShapeShiftedValue()) {
        llvm::SmallVector<mlir::Value> shapeShiftOperands;

        size_t shapeIdx = 0;
        for (auto [startIndex, extent] :
             llvm::zip_equal(targetShapeCreationInfo.startIndices,
                             targetShapeCreationInfo.extents)) {
          shapeShiftOperands.push_back(
              Fortran::lower::omp::internal::mapTemporaryValue(
                  builder, targetOp, startIndex,
                  liveInName + ".start_idx.dim" + std::to_string(shapeIdx)));
          shapeShiftOperands.push_back(
              Fortran::lower::omp::internal::mapTemporaryValue(
                  builder, targetOp, extent,
                  liveInName + ".extent.dim" + std::to_string(shapeIdx)));
          ++shapeIdx;
        }

        auto shapeShiftType = fir::ShapeShiftType::get(
            builder.getContext(), shapeShiftOperands.size() / 2);
        return builder.create<fir::ShapeShiftOp>(
            liveInArg.getLoc(), shapeShiftType, shapeShiftOperands);
      }

      llvm::SmallVector<mlir::Value> shapeOperands;
      size_t shapeIdx = 0;
      for (auto extent : targetShapeCreationInfo.extents) {
        shapeOperands.push_back(
            Fortran::lower::omp::internal::mapTemporaryValue(
                builder, targetOp, extent,
                liveInName + ".extent.dim" + std::to_string(shapeIdx)));
        ++shapeIdx;
      }

      return builder.create<fir::ShapeOp>(liveInArg.getLoc(), shapeOperands);
    }();

    return builder.create<hlfir::DeclareOp>(liveInArg.getLoc(), liveInArg,
                                            liveInName, shape);
  }

  mlir::omp::TeamsOp
  genTeamsOp(mlir::Location loc,
             mlir::ConversionPatternRewriter &rewriter) const {
    auto teamsOp = rewriter.create<mlir::omp::TeamsOp>(
        loc, /*clauses=*/mlir::omp::TeamsOperands{});

    rewriter.createBlock(&teamsOp.getRegion());
    rewriter.setInsertionPoint(rewriter.create<mlir::omp::TerminatorOp>(loc));

    return teamsOp;
  }

  void genLoopNestClauseOps(
      mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
      looputils::LoopNestToIndVarMap &loopNest, mlir::IRMapping &mapper,
      mlir::omp::LoopNestOperands &loopNestClauseOps,
      mlir::omp::TargetOperands *targetClauseOps = nullptr) const {
    assert(loopNestClauseOps.loopLowerBounds.empty() &&
           "Loop nest bounds were already emitted!");

    // Clones the chain of ops defining a certain loop bound or its step into
    // the parallel region. For example, if the value of a bound is defined by a
    // `fir.convert`op, this lambda clones the `fir.convert` as well as the
    // value it converts from. We do this since `omp.target` regions are
    // isolated from above.
    auto cloneBoundOrStepOpChain =
        [&](mlir::Operation *operation) -> mlir::Operation * {
      llvm::SetVector<mlir::Operation *> opChain;
      looputils::collectIndirectConstOpChain(operation, opChain);

      mlir::Operation *result;
      for (mlir::Operation *link : opChain)
        result = rewriter.clone(*link, mapper);

      return result;
    };

    auto hostEvalCapture = [&](mlir::Value var,
                               llvm::SmallVectorImpl<mlir::Value> &bounds) {
      var = cloneBoundOrStepOpChain(var.getDefiningOp())->getResult(0);
      bounds.push_back(var);

      if (targetClauseOps)
        targetClauseOps->hostEvalVars.push_back(var);
    };

    for (auto &[doLoop, _] : loopNest) {
      hostEvalCapture(doLoop.getLowerBound(),
                      loopNestClauseOps.loopLowerBounds);
      hostEvalCapture(doLoop.getUpperBound(),
                      loopNestClauseOps.loopUpperBounds);
      hostEvalCapture(doLoop.getStep(), loopNestClauseOps.loopSteps);
    }

    loopNestClauseOps.loopInclusive = rewriter.getUnitAttr();
  }

  mlir::omp::DistributeOp
  genDistributeOp(mlir::Location loc,
                  mlir::ConversionPatternRewriter &rewriter) const {
    auto distOp = rewriter.create<mlir::omp::DistributeOp>(
        loc, /*clauses=*/mlir::omp::DistributeOperands{});

    rewriter.createBlock(&distOp.getRegion());
    return distOp;
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

    if (func.isDeclaration()) {
      return;
    }

    auto *context = &getContext();

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
    target
        .addLegalDialect<fir::FIROpsDialect, hlfir::hlfirDialect,
                         mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         mlir::omp::OpenMPDialect, mlir::cf::ControlFlowDialect,
                         mlir::math::MathDialect, mlir::LLVM::LLVMDialect>();

    target.addDynamicallyLegalOp<fir::DoLoopOp>([&](fir::DoLoopOp op) {
      return !op.getUnordered() || concurrentLoopsToSkip.contains(op);
    });

    if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                               std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting do-concurrent op");
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
