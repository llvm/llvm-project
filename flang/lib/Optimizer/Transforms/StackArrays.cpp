//===- StackArrays.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/LowLevelIntrinsics.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace fir {
#define GEN_PASS_DEF_STACKARRAYS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "stack-arrays"

static llvm::cl::opt<std::size_t> maxAllocsPerFunc(
    "stack-arrays-max-allocs",
    llvm::cl::desc("The maximum number of heap allocations to consider in one "
                   "function before skipping (to save compilation time). Set "
                   "to 0 for no limit."),
    llvm::cl::init(1000), llvm::cl::Hidden);

namespace {

/// The state of an SSA value at each program point
enum class AllocationState {
  /// This means that the allocation state of a variable cannot be determined
  /// at this program point, e.g. because one route through a conditional freed
  /// the variable and the other route didn't.
  /// This asserts a known-unknown: different from the unknown-unknown of having
  /// no AllocationState stored for a particular SSA value
  Unknown,
  /// Means this SSA value was allocated on the heap in this function and has
  /// now been freed
  Freed,
  /// Means this SSA value was allocated on the heap in this function and is a
  /// candidate for moving to the stack
  Allocated,
};

/// Stores where an alloca should be inserted. If the PointerUnion is an
/// Operation the alloca should be inserted /after/ the operation. If it is a
/// block, the alloca can be placed anywhere in that block.
class InsertionPoint {
  llvm::PointerUnion<mlir::Operation *, mlir::Block *> location;
  bool saveRestoreStack;

  /// Get contained pointer type or nullptr
  template <class T>
  T *tryGetPtr() const {
    // Use llvm::dyn_cast_if_present because location may be null here.
    if (T *ptr = llvm::dyn_cast_if_present<T *>(location))
      return ptr;
    return nullptr;
  }

public:
  template <class T>
  InsertionPoint(T *ptr, bool saveRestoreStack = false)
      : location(ptr), saveRestoreStack{saveRestoreStack} {}
  InsertionPoint(std::nullptr_t null)
      : location(null), saveRestoreStack{false} {}

  /// Get contained operation, or nullptr
  mlir::Operation *tryGetOperation() const {
    return tryGetPtr<mlir::Operation>();
  }

  /// Get contained block, or nullptr
  mlir::Block *tryGetBlock() const { return tryGetPtr<mlir::Block>(); }

  /// Get whether the stack should be saved/restored. If yes, an llvm.stacksave
  /// intrinsic should be added before the alloca, and an llvm.stackrestore
  /// intrinsic should be added where the freemem is
  bool shouldSaveRestoreStack() const { return saveRestoreStack; }

  operator bool() const { return tryGetOperation() || tryGetBlock(); }

  bool operator==(const InsertionPoint &rhs) const {
    return (location == rhs.location) &&
           (saveRestoreStack == rhs.saveRestoreStack);
  }

  bool operator!=(const InsertionPoint &rhs) const { return !(*this == rhs); }
};

/// Maps SSA values to their AllocationState at a particular program point.
/// Also caches the insertion points for the new alloca operations
class LatticePoint : public mlir::dataflow::AbstractDenseLattice {
  // Maps all values we are interested in to states
  llvm::SmallDenseMap<mlir::Value, AllocationState, 1> stateMap;

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LatticePoint)
  using AbstractDenseLattice::AbstractDenseLattice;

  bool operator==(const LatticePoint &rhs) const {
    return stateMap == rhs.stateMap;
  }

  /// Join the lattice accross control-flow edges
  mlir::ChangeResult join(const AbstractDenseLattice &lattice) override;

  void print(llvm::raw_ostream &os) const override;

  /// Clear all modifications
  mlir::ChangeResult reset();

  /// Set the state of an SSA value
  mlir::ChangeResult set(mlir::Value value, AllocationState state);

  /// Get fir.allocmem ops which were allocated in this function and always
  /// freed before the function returns, plus whre to insert replacement
  /// fir.alloca ops
  void appendFreedValues(llvm::DenseSet<mlir::Value> &out) const;

  std::optional<AllocationState> get(mlir::Value val) const;
};

class AllocationAnalysis
    : public mlir::dataflow::DenseForwardDataFlowAnalysis<LatticePoint> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

  mlir::LogicalResult visitOperation(mlir::Operation *op,
                                     const LatticePoint &before,
                                     LatticePoint *after) override;

  /// At an entry point, the last modifications of all memory resources are
  /// yet to be determined
  void setToEntryState(LatticePoint *lattice) override;

protected:
  /// Visit control flow operations and decide whether to call visitOperation
  /// to apply the transfer function
  mlir::LogicalResult processOperation(mlir::Operation *op) override;
};

/// Drives analysis to find candidate fir.allocmem operations which could be
/// moved to the stack. Intended to be used with mlir::Pass::getAnalysis
class StackArraysAnalysisWrapper {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StackArraysAnalysisWrapper)

  // Maps fir.allocmem -> place to insert alloca
  using AllocMemMap = llvm::DenseMap<mlir::Operation *, InsertionPoint>;

  StackArraysAnalysisWrapper(mlir::Operation *op) {}

  // returns nullptr if analysis failed
  const AllocMemMap *getCandidateOps(mlir::Operation *func);

private:
  llvm::DenseMap<mlir::Operation *, AllocMemMap> funcMaps;

  llvm::LogicalResult analyseFunction(mlir::Operation *func);
};

/// Converts a fir.allocmem to a fir.alloca
class AllocMemConversion : public mlir::OpRewritePattern<fir::AllocMemOp> {
public:
  explicit AllocMemConversion(
      mlir::MLIRContext *ctx,
      const StackArraysAnalysisWrapper::AllocMemMap &candidateOps)
      : OpRewritePattern(ctx), candidateOps{candidateOps} {}

  llvm::LogicalResult
  matchAndRewrite(fir::AllocMemOp allocmem,
                  mlir::PatternRewriter &rewriter) const override;

  /// Determine where to insert the alloca operation. The returned value should
  /// be checked to see if it is inside a loop
  static InsertionPoint findAllocaInsertionPoint(fir::AllocMemOp &oldAlloc);

private:
  /// Handle to the DFA (already run)
  const StackArraysAnalysisWrapper::AllocMemMap &candidateOps;

  /// If we failed to find an insertion point not inside a loop, see if it would
  /// be safe to use an llvm.stacksave/llvm.stackrestore inside the loop
  static InsertionPoint findAllocaLoopInsertionPoint(fir::AllocMemOp &oldAlloc);

  /// Returns the alloca if it was successfully inserted, otherwise {}
  std::optional<fir::AllocaOp>
  insertAlloca(fir::AllocMemOp &oldAlloc,
               mlir::PatternRewriter &rewriter) const;

  /// Inserts a stacksave before oldAlloc and a stackrestore after each freemem
  void insertStackSaveRestore(fir::AllocMemOp &oldAlloc,
                              mlir::PatternRewriter &rewriter) const;
};

class StackArraysPass : public fir::impl::StackArraysBase<StackArraysPass> {
public:
  StackArraysPass() = default;
  StackArraysPass(const StackArraysPass &pass);

  llvm::StringRef getDescription() const override;

  void runOnOperation() override;

private:
  Statistic runCount{this, "stackArraysRunCount",
                     "Number of heap allocations moved to the stack"};
};

} // namespace

static void print(llvm::raw_ostream &os, AllocationState state) {
  switch (state) {
  case AllocationState::Unknown:
    os << "Unknown";
    break;
  case AllocationState::Freed:
    os << "Freed";
    break;
  case AllocationState::Allocated:
    os << "Allocated";
    break;
  }
}

/// Join two AllocationStates for the same value coming from different CFG
/// blocks
static AllocationState join(AllocationState lhs, AllocationState rhs) {
  //           | Allocated | Freed     | Unknown
  // ========= | ========= | ========= | =========
  // Allocated | Allocated | Unknown   | Unknown
  // Freed     | Unknown   | Freed     | Unknown
  // Unknown   | Unknown   | Unknown   | Unknown
  if (lhs == rhs)
    return lhs;
  return AllocationState::Unknown;
}

mlir::ChangeResult LatticePoint::join(const AbstractDenseLattice &lattice) {
  const auto &rhs = static_cast<const LatticePoint &>(lattice);
  mlir::ChangeResult changed = mlir::ChangeResult::NoChange;

  // add everything from rhs to map, handling cases where values are in both
  for (const auto &[value, rhsState] : rhs.stateMap) {
    auto it = stateMap.find(value);
    if (it != stateMap.end()) {
      // value is present in both maps
      AllocationState myState = it->second;
      AllocationState newState = ::join(myState, rhsState);
      if (newState != myState) {
        changed = mlir::ChangeResult::Change;
        it->getSecond() = newState;
      }
    } else {
      // value not present in current map: add it
      stateMap.insert({value, rhsState});
      changed = mlir::ChangeResult::Change;
    }
  }

  return changed;
}

void LatticePoint::print(llvm::raw_ostream &os) const {
  for (const auto &[value, state] : stateMap) {
    os << "\n * " << value << ": ";
    ::print(os, state);
  }
}

mlir::ChangeResult LatticePoint::reset() {
  if (stateMap.empty())
    return mlir::ChangeResult::NoChange;
  stateMap.clear();
  return mlir::ChangeResult::Change;
}

mlir::ChangeResult LatticePoint::set(mlir::Value value, AllocationState state) {
  if (stateMap.count(value)) {
    // already in map
    AllocationState &oldState = stateMap[value];
    if (oldState != state) {
      stateMap[value] = state;
      return mlir::ChangeResult::Change;
    }
    return mlir::ChangeResult::NoChange;
  }
  stateMap.insert({value, state});
  return mlir::ChangeResult::Change;
}

/// Get values which were allocated in this function and always freed before
/// the function returns
void LatticePoint::appendFreedValues(llvm::DenseSet<mlir::Value> &out) const {
  for (auto &[value, state] : stateMap) {
    if (state == AllocationState::Freed)
      out.insert(value);
  }
}

std::optional<AllocationState> LatticePoint::get(mlir::Value val) const {
  auto it = stateMap.find(val);
  if (it == stateMap.end())
    return {};
  return it->second;
}

static mlir::Value lookThroughDeclaresAndConverts(mlir::Value value) {
  while (mlir::Operation *op = value.getDefiningOp()) {
    if (auto declareOp = llvm::dyn_cast<fir::DeclareOp>(op))
      value = declareOp.getMemref();
    else if (auto convertOp = llvm::dyn_cast<fir::ConvertOp>(op))
      value = convertOp->getOperand(0);
    else
      return value;
  }
  return value;
}

mlir::LogicalResult AllocationAnalysis::visitOperation(
    mlir::Operation *op, const LatticePoint &before, LatticePoint *after) {
  LLVM_DEBUG(llvm::dbgs() << "StackArrays: Visiting operation: " << *op
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "--Lattice in: " << before << "\n");

  // propagate before -> after
  mlir::ChangeResult changed = after->join(before);

  if (auto allocmem = mlir::dyn_cast<fir::AllocMemOp>(op)) {
    assert(op->getNumResults() == 1 && "fir.allocmem has one result");
    auto attr = op->getAttrOfType<fir::MustBeHeapAttr>(
        fir::MustBeHeapAttr::getAttrName());
    if (attr && attr.getValue()) {
      LLVM_DEBUG(llvm::dbgs() << "--Found fir.must_be_heap: skipping\n");
      // skip allocation marked not to be moved
      return mlir::success();
    }

    auto retTy = allocmem.getAllocatedType();
    if (!mlir::isa<fir::SequenceType>(retTy)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "--Allocation is not for an array: skipping\n");
      return mlir::success();
    }

    mlir::Value result = op->getResult(0);
    changed |= after->set(result, AllocationState::Allocated);
  } else if (mlir::isa<fir::FreeMemOp>(op)) {
    assert(op->getNumOperands() == 1 && "fir.freemem has one operand");
    mlir::Value operand = op->getOperand(0);

    // Note: StackArrays is scheduled in the pass pipeline after lowering hlfir
    // to fir. Therefore, we only need to handle `fir::DeclareOp`s. Also look
    // past converts in case the pointer was changed between different pointer
    // types.
    operand = lookThroughDeclaresAndConverts(operand);

    std::optional<AllocationState> operandState = before.get(operand);
    if (operandState && *operandState == AllocationState::Allocated) {
      // don't tag things not allocated in this function as freed, so that we
      // don't think they are candidates for moving to the stack
      changed |= after->set(operand, AllocationState::Freed);
    }
  } else if (mlir::isa<fir::ResultOp>(op)) {
    mlir::Operation *parent = op->getParentOp();
    LatticePoint *parentLattice = getLattice(getProgramPointAfter(parent));
    assert(parentLattice);
    mlir::ChangeResult parentChanged = parentLattice->join(*after);
    propagateIfChanged(parentLattice, parentChanged);
  }

  // we pass lattices straight through fir.call because called functions should
  // not deallocate flang-generated array temporaries

  LLVM_DEBUG(llvm::dbgs() << "--Lattice out: " << *after << "\n");
  propagateIfChanged(after, changed);
  return mlir::success();
}

void AllocationAnalysis::setToEntryState(LatticePoint *lattice) {
  propagateIfChanged(lattice, lattice->reset());
}

/// Mostly a copy of AbstractDenseLattice::processOperation - the difference
/// being that call operations are passed through to the transfer function
mlir::LogicalResult AllocationAnalysis::processOperation(mlir::Operation *op) {
  mlir::ProgramPoint *point = getProgramPointAfter(op);
  // If the containing block is not executable, bail out.
  if (op->getBlock() != nullptr &&
      !getOrCreateFor<mlir::dataflow::Executable>(
           point, getProgramPointBefore(op->getBlock()))
           ->isLive())
    return mlir::success();

  // Get the dense lattice to update
  mlir::dataflow::AbstractDenseLattice *after = getLattice(point);

  // If this op implements region control-flow, then control-flow dictates its
  // transfer function.
  if (auto branch = mlir::dyn_cast<mlir::RegionBranchOpInterface>(op)) {
    visitRegionBranchOperation(point, branch, after);
    return mlir::success();
  }

  // pass call operations through to the transfer function

  // Get the dense state before the execution of the op.
  const mlir::dataflow::AbstractDenseLattice *before =
      getLatticeFor(point, getProgramPointBefore(op));

  /// Invoke the operation transfer function
  return visitOperationImpl(op, *before, after);
}

llvm::LogicalResult
StackArraysAnalysisWrapper::analyseFunction(mlir::Operation *func) {
  assert(mlir::isa<mlir::func::FuncOp>(func));
  size_t nAllocs = 0;
  func->walk([&nAllocs](fir::AllocMemOp) { nAllocs++; });
  // don't bother with the analysis if there are no heap allocations
  if (nAllocs == 0)
    return mlir::success();
  if ((maxAllocsPerFunc != 0) && (nAllocs > maxAllocsPerFunc)) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping stack arrays for function with "
                            << nAllocs << " heap allocations");
    return mlir::success();
  }

  mlir::DataFlowSolver solver;
  // constant propagation is required for dead code analysis, dead code analysis
  // is required to mark blocks live (required for mlir dense dfa)
  solver.load<mlir::dataflow::SparseConstantPropagation>();
  solver.load<mlir::dataflow::DeadCodeAnalysis>();

  auto [it, inserted] = funcMaps.try_emplace(func);
  AllocMemMap &candidateOps = it->second;

  solver.load<AllocationAnalysis>();
  if (failed(solver.initializeAndRun(func))) {
    llvm::errs() << "DataFlowSolver failed!";
    return mlir::failure();
  }

  LatticePoint point{solver.getProgramPointAfter(func)};
  auto joinOperationLattice = [&](mlir::Operation *op) {
    const LatticePoint *lattice =
        solver.lookupState<LatticePoint>(solver.getProgramPointAfter(op));
    // there will be no lattice for an unreachable block
    if (lattice)
      (void)point.join(*lattice);
  };

  func->walk([&](mlir::func::ReturnOp child) { joinOperationLattice(child); });
  func->walk([&](fir::UnreachableOp child) { joinOperationLattice(child); });
  func->walk(
      [&](mlir::omp::TerminatorOp child) { joinOperationLattice(child); });
  func->walk([&](mlir::omp::YieldOp child) { joinOperationLattice(child); });

  llvm::DenseSet<mlir::Value> freedValues;
  point.appendFreedValues(freedValues);

  // We only replace allocations which are definately freed on all routes
  // through the function because otherwise the allocation may have an intende
  // lifetime longer than the current stack frame (e.g. a heap allocation which
  // is then freed by another function).
  for (mlir::Value freedValue : freedValues) {
    fir::AllocMemOp allocmem = freedValue.getDefiningOp<fir::AllocMemOp>();
    InsertionPoint insertionPoint =
        AllocMemConversion::findAllocaInsertionPoint(allocmem);
    if (insertionPoint)
      candidateOps.insert({allocmem, insertionPoint});
  }

  LLVM_DEBUG(for (auto [allocMemOp, _]
                  : candidateOps) {
    llvm::dbgs() << "StackArrays: Found candidate op: " << *allocMemOp << '\n';
  });
  return mlir::success();
}

const StackArraysAnalysisWrapper::AllocMemMap *
StackArraysAnalysisWrapper::getCandidateOps(mlir::Operation *func) {
  if (!funcMaps.contains(func))
    if (mlir::failed(analyseFunction(func)))
      return nullptr;
  return &funcMaps[func];
}

/// Restore the old allocation type exected by existing code
static mlir::Value convertAllocationType(mlir::PatternRewriter &rewriter,
                                         const mlir::Location &loc,
                                         mlir::Value heap, mlir::Value stack) {
  mlir::Type heapTy = heap.getType();
  mlir::Type stackTy = stack.getType();

  if (heapTy == stackTy)
    return stack;

  fir::HeapType firHeapTy = mlir::cast<fir::HeapType>(heapTy);
  LLVM_ATTRIBUTE_UNUSED fir::ReferenceType firRefTy =
      mlir::cast<fir::ReferenceType>(stackTy);
  assert(firHeapTy.getElementType() == firRefTy.getElementType() &&
         "Allocations must have the same type");

  auto insertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointAfter(stack.getDefiningOp());
  mlir::Value conv =
      rewriter.create<fir::ConvertOp>(loc, firHeapTy, stack).getResult();
  rewriter.restoreInsertionPoint(insertionPoint);
  return conv;
}

llvm::LogicalResult
AllocMemConversion::matchAndRewrite(fir::AllocMemOp allocmem,
                                    mlir::PatternRewriter &rewriter) const {
  auto oldInsertionPt = rewriter.saveInsertionPoint();
  // add alloca operation
  std::optional<fir::AllocaOp> alloca = insertAlloca(allocmem, rewriter);
  rewriter.restoreInsertionPoint(oldInsertionPt);
  if (!alloca)
    return mlir::failure();

  // remove freemem operations
  llvm::SmallVector<mlir::Operation *> erases;
  mlir::Operation *parent = allocmem->getParentOp();
  // TODO: this shouldn't need to be re-calculated for every allocmem
  parent->walk([&](fir::FreeMemOp freeOp) {
    if (lookThroughDeclaresAndConverts(freeOp->getOperand(0)) == allocmem)
      erases.push_back(freeOp);
  });

  // now we are done iterating the users, it is safe to mutate them
  for (mlir::Operation *erase : erases)
    rewriter.eraseOp(erase);

  // replace references to heap allocation with references to stack allocation
  mlir::Value newValue = convertAllocationType(
      rewriter, allocmem.getLoc(), allocmem.getResult(), alloca->getResult());
  rewriter.replaceAllUsesWith(allocmem.getResult(), newValue);

  // remove allocmem operation
  rewriter.eraseOp(allocmem.getOperation());

  return mlir::success();
}

static bool isInLoop(mlir::Block *block) {
  return mlir::LoopLikeOpInterface::blockIsInLoop(block);
}

static bool isInLoop(mlir::Operation *op) {
  return isInLoop(op->getBlock()) ||
         op->getParentOfType<mlir::LoopLikeOpInterface>();
}

InsertionPoint
AllocMemConversion::findAllocaInsertionPoint(fir::AllocMemOp &oldAlloc) {
  // Ideally the alloca should be inserted at the end of the function entry
  // block so that we do not allocate stack space in a loop. However,
  // the operands to the alloca may not be available that early, so insert it
  // after the last operand becomes available
  // If the old allocmem op was in an openmp region then it should not be moved
  // outside of that
  LLVM_DEBUG(llvm::dbgs() << "StackArrays: findAllocaInsertionPoint: "
                          << oldAlloc << "\n");

  // check that an Operation or Block we are about to return is not in a loop
  auto checkReturn = [&](auto *point) -> InsertionPoint {
    if (isInLoop(point)) {
      mlir::Operation *oldAllocOp = oldAlloc.getOperation();
      if (isInLoop(oldAllocOp)) {
        // where we want to put it is in a loop, and even the old location is in
        // a loop. Give up.
        return findAllocaLoopInsertionPoint(oldAlloc);
      }
      return {oldAllocOp};
    }
    return {point};
  };

  auto oldOmpRegion =
      oldAlloc->getParentOfType<mlir::omp::OutlineableOpenMPOpInterface>();

  // Find when the last operand value becomes available
  mlir::Block *operandsBlock = nullptr;
  mlir::Operation *lastOperand = nullptr;
  for (mlir::Value operand : oldAlloc.getOperands()) {
    LLVM_DEBUG(llvm::dbgs() << "--considering operand " << operand << "\n");
    mlir::Operation *op = operand.getDefiningOp();
    if (!op)
      return checkReturn(oldAlloc.getOperation());
    if (!operandsBlock)
      operandsBlock = op->getBlock();
    else if (operandsBlock != op->getBlock()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "----operand declared in a different block!\n");
      // Operation::isBeforeInBlock requires the operations to be in the same
      // block. The best we can do is the location of the allocmem.
      return checkReturn(oldAlloc.getOperation());
    }
    if (!lastOperand || lastOperand->isBeforeInBlock(op))
      lastOperand = op;
  }

  if (lastOperand) {
    // there were value operands to the allocmem so insert after the last one
    LLVM_DEBUG(llvm::dbgs()
               << "--Placing after last operand: " << *lastOperand << "\n");
    // check we aren't moving out of an omp region
    auto lastOpOmpRegion =
        lastOperand->getParentOfType<mlir::omp::OutlineableOpenMPOpInterface>();
    if (lastOpOmpRegion == oldOmpRegion)
      return checkReturn(lastOperand);
    // Presumably this happened because the operands became ready before the
    // start of this openmp region. (lastOpOmpRegion != oldOmpRegion) should
    // imply that oldOmpRegion comes after lastOpOmpRegion.
    return checkReturn(oldOmpRegion.getAllocaBlock());
  }

  // There were no value operands to the allocmem so we are safe to insert it
  // as early as we want

  // handle openmp case
  if (oldOmpRegion)
    return checkReturn(oldOmpRegion.getAllocaBlock());

  // fall back to the function entry block
  mlir::func::FuncOp func = oldAlloc->getParentOfType<mlir::func::FuncOp>();
  assert(func && "This analysis is run on func.func");
  mlir::Block &entryBlock = func.getBlocks().front();
  LLVM_DEBUG(llvm::dbgs() << "--Placing at the start of func entry block\n");
  return checkReturn(&entryBlock);
}

InsertionPoint
AllocMemConversion::findAllocaLoopInsertionPoint(fir::AllocMemOp &oldAlloc) {
  mlir::Operation *oldAllocOp = oldAlloc;
  // This is only called as a last resort. We should try to insert at the
  // location of the old allocation, which is inside of a loop, using
  // llvm.stacksave/llvm.stackrestore

  // find freemem ops
  llvm::SmallVector<mlir::Operation *, 1> freeOps;

  for (mlir::Operation *user : oldAllocOp->getUsers()) {
    if (auto declareOp = mlir::dyn_cast_if_present<fir::DeclareOp>(user)) {
      for (mlir::Operation *user : declareOp->getUsers()) {
        if (mlir::isa<fir::FreeMemOp>(user))
          freeOps.push_back(user);
      }
    }

    if (mlir::isa<fir::FreeMemOp>(user))
      freeOps.push_back(user);
  }

  assert(freeOps.size() && "DFA should only return freed memory");

  // Don't attempt to reason about a stacksave/stackrestore between different
  // blocks
  for (mlir::Operation *free : freeOps)
    if (free->getBlock() != oldAllocOp->getBlock())
      return {nullptr};

  // Check that there aren't any other stack allocations in between the
  // stack save and stack restore
  // note: for flang generated temporaries there should only be one free op
  for (mlir::Operation *free : freeOps) {
    for (mlir::Operation *op = oldAlloc; op && op != free;
         op = op->getNextNode()) {
      if (mlir::isa<fir::AllocaOp>(op))
        return {nullptr};
    }
  }

  return InsertionPoint{oldAllocOp, /*shouldStackSaveRestore=*/true};
}

std::optional<fir::AllocaOp>
AllocMemConversion::insertAlloca(fir::AllocMemOp &oldAlloc,
                                 mlir::PatternRewriter &rewriter) const {
  auto it = candidateOps.find(oldAlloc.getOperation());
  if (it == candidateOps.end())
    return {};
  InsertionPoint insertionPoint = it->second;
  if (!insertionPoint)
    return {};

  if (insertionPoint.shouldSaveRestoreStack())
    insertStackSaveRestore(oldAlloc, rewriter);

  mlir::Location loc = oldAlloc.getLoc();
  mlir::Type varTy = oldAlloc.getInType();
  if (mlir::Operation *op = insertionPoint.tryGetOperation()) {
    rewriter.setInsertionPointAfter(op);
  } else {
    mlir::Block *block = insertionPoint.tryGetBlock();
    assert(block && "There must be a valid insertion point");
    rewriter.setInsertionPointToStart(block);
  }

  auto unpackName = [](std::optional<llvm::StringRef> opt) -> llvm::StringRef {
    if (opt)
      return *opt;
    return {};
  };

  llvm::StringRef uniqName = unpackName(oldAlloc.getUniqName());
  llvm::StringRef bindcName = unpackName(oldAlloc.getBindcName());
  return rewriter.create<fir::AllocaOp>(loc, varTy, uniqName, bindcName,
                                        oldAlloc.getTypeparams(),
                                        oldAlloc.getShape());
}

void AllocMemConversion::insertStackSaveRestore(
    fir::AllocMemOp &oldAlloc, mlir::PatternRewriter &rewriter) const {
  auto oldPoint = rewriter.saveInsertionPoint();
  auto mod = oldAlloc->getParentOfType<mlir::ModuleOp>();
  fir::FirOpBuilder builder{rewriter, mod};

  builder.setInsertionPoint(oldAlloc);
  mlir::Value sp = builder.genStackSave(oldAlloc.getLoc());

  auto createStackRestoreCall = [&](mlir::Operation *user) {
    builder.setInsertionPoint(user);
    builder.genStackRestore(user->getLoc(), sp);
  };

  for (mlir::Operation *user : oldAlloc->getUsers()) {
    if (auto declareOp = mlir::dyn_cast_if_present<fir::DeclareOp>(user)) {
      for (mlir::Operation *user : declareOp->getUsers()) {
        if (mlir::isa<fir::FreeMemOp>(user))
          createStackRestoreCall(user);
      }
    }

    if (mlir::isa<fir::FreeMemOp>(user)) {
      createStackRestoreCall(user);
    }
  }

  rewriter.restoreInsertionPoint(oldPoint);
}

StackArraysPass::StackArraysPass(const StackArraysPass &pass)
    : fir::impl::StackArraysBase<StackArraysPass>(pass) {}

llvm::StringRef StackArraysPass::getDescription() const {
  return "Move heap allocated array temporaries to the stack";
}

void StackArraysPass::runOnOperation() {
  mlir::func::FuncOp func = getOperation();

  auto &analysis = getAnalysis<StackArraysAnalysisWrapper>();
  const StackArraysAnalysisWrapper::AllocMemMap *candidateOps =
      analysis.getCandidateOps(func);
  if (!candidateOps) {
    signalPassFailure();
    return;
  }

  if (candidateOps->empty())
    return;
  runCount += candidateOps->size();

  llvm::SmallVector<mlir::Operation *> opsToConvert;
  opsToConvert.reserve(candidateOps->size());
  for (auto [op, _] : *candidateOps)
    opsToConvert.push_back(op);

  mlir::MLIRContext &context = getContext();
  mlir::RewritePatternSet patterns(&context);
  mlir::GreedyRewriteConfig config;
  // prevent the pattern driver form merging blocks
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;

  patterns.insert<AllocMemConversion>(&context, *candidateOps);
  if (mlir::failed(mlir::applyOpPatternsGreedily(
          opsToConvert, std::move(patterns), config))) {
    mlir::emitError(func->getLoc(), "error in stack arrays optimization\n");
    signalPassFailure();
  }
}
