//===-- ArrayValueCopy.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Factory.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-array-value-copy"

using namespace fir;

using OperationUseMapT = llvm::DenseMap<mlir::Operation *, mlir::Operation *>;

namespace {

/// Array copy analysis.
/// Perform an interference analysis between array values.
///
/// Lowering will generate a sequence of the following form.
/// ```mlir
///   %a_1 = fir.array_load %array_1(%shape) : ...
///   ...
///   %a_j = fir.array_load %array_j(%shape) : ...
///   ...
///   %a_n = fir.array_load %array_n(%shape) : ...
///     ...
///     %v_i = fir.array_fetch %a_i, ...
///     %a_j1 = fir.array_update %a_j, ...
///     ...
///   fir.array_merge_store %a_j, %a_jn to %array_j : ...
/// ```
///
/// The analysis is to determine if there are any conflicts. A conflict is when
/// one the following cases occurs.
///
/// 1. There is an `array_update` to an array value, a_j, such that a_j was
/// loaded from the same array memory reference (array_j) but with a different
/// shape as the other array values a_i, where i != j. [Possible overlapping
/// arrays.]
///
/// 2. There is either an array_fetch or array_update of a_j with a different
/// set of index values. [Possible loop-carried dependence.]
///
/// If none of the array values overlap in storage and the accesses are not
/// loop-carried, then the arrays are conflict-free and no copies are required.
class ArrayCopyAnalysis {
public:
  using ConflictSetT = llvm::SmallPtrSet<mlir::Operation *, 16>;
  using UseSetT = llvm::SmallPtrSet<mlir::OpOperand *, 8>;
  using LoadMapSetsT =
      llvm::DenseMap<mlir::Operation *, SmallVector<Operation *>>;

  ArrayCopyAnalysis(mlir::Operation *op) : operation{op} { construct(op); }

  mlir::Operation *getOperation() const { return operation; }

  /// Return true iff the `array_merge_store` has potential conflicts.
  bool hasPotentialConflict(mlir::Operation *op) const {
    LLVM_DEBUG(llvm::dbgs()
               << "looking for a conflict on " << *op
               << " and the set has a total of " << conflicts.size() << '\n');
    return conflicts.contains(op);
  }

  /// Return the use map. The use map maps array fetch and update operations
  /// back to the array load that is the original source of the array value.
  const OperationUseMapT &getUseMap() const { return useMap; }

  /// Find all the array operations that access the array value that is loaded
  /// by the array load operation, `load`.
  const llvm::SmallVector<mlir::Operation *> &arrayAccesses(ArrayLoadOp load);

private:
  void construct(mlir::Operation *topLevelOp);

  mlir::Operation *operation; // operation that analysis ran upon
  ConflictSetT conflicts;     // set of conflicts (loads and merge stores)
  OperationUseMapT useMap;
  LoadMapSetsT loadMapSets;
};
} // namespace

namespace {
/// Helper class to collect all array operations that produced an array value.
class ReachCollector {
private:
  // If provided, the `loopRegion` is the body of a loop that produces the array
  // of interest.
  ReachCollector(llvm::SmallVectorImpl<mlir::Operation *> &reach,
                 mlir::Region *loopRegion)
      : reach{reach}, loopRegion{loopRegion} {}

  void collectArrayAccessFrom(mlir::Operation *op, mlir::ValueRange range) {
    llvm::errs() << "COLLECT " << *op << "\n";
    if (range.empty()) {
      collectArrayAccessFrom(op, mlir::Value{});
      return;
    }
    for (mlir::Value v : range)
      collectArrayAccessFrom(v);
  }

  // TODO: Replace recursive algorithm on def-use chain with an iterative one
  // with an explicit stack.
  void collectArrayAccessFrom(mlir::Operation *op, mlir::Value val) {
    // `val` is defined by an Op, process the defining Op.
    // If `val` is defined by a region containing Op, we want to drill down
    // and through that Op's region(s).
    llvm::errs() << "COLLECT " << *op << "\n";
    LLVM_DEBUG(llvm::dbgs() << "popset: " << *op << '\n');
    auto popFn = [&](auto rop) {
      assert(val && "op must have a result value");
      auto resNum = val.cast<mlir::OpResult>().getResultNumber();
      llvm::SmallVector<mlir::Value> results;
      rop.resultToSourceOps(results, resNum);
      for (auto u : results)
        collectArrayAccessFrom(u);
    };
    if (auto rop = mlir::dyn_cast<fir::DoLoopOp>(op)) {
      popFn(rop);
      return;
    }
    if (auto rop = mlir::dyn_cast<fir::IfOp>(op)) {
      popFn(rop);
      return;
    }
    if (auto mergeStore = mlir::dyn_cast<ArrayMergeStoreOp>(op)) {
      if (opIsInsideLoops(mergeStore))
        collectArrayAccessFrom(mergeStore.sequence());
      return;
    }

    if (mlir::isa<AllocaOp, AllocMemOp>(op)) {
      // Look for any stores inside the loops, and collect an array operation
      // that produced the value being stored to it.
      for (mlir::Operation *user : op->getUsers())
        if (auto store = mlir::dyn_cast<fir::StoreOp>(user))
          if (opIsInsideLoops(store))
            collectArrayAccessFrom(store.value());
      return;
    }

    // Otherwise, Op does not contain a region so just chase its operands.
    if (mlir::isa<ArrayLoadOp, ArrayUpdateOp, ArrayModifyOp, ArrayFetchOp>(
            op)) {
      LLVM_DEBUG(llvm::dbgs() << "add " << *op << " to reachable set\n");
      reach.emplace_back(op);
    }
    // Array modify assignment is performed on the result. So the analysis
    // must look at the what is done with the result.
    if (mlir::isa<ArrayModifyOp>(op))
      for (mlir::Operation *user : op->getResult(0).getUsers())
        followUsers(user);

    for (auto u : op->getOperands())
      collectArrayAccessFrom(u);
  }

  void collectArrayAccessFrom(mlir::BlockArgument ba) {
    auto *parent = ba.getOwner()->getParentOp();
    // If inside an Op holding a region, the block argument corresponds to an
    // argument passed to the containing Op.
    auto popFn = [&](auto rop) {
      collectArrayAccessFrom(rop.blockArgToSourceOp(ba.getArgNumber()));
    };
    if (auto rop = mlir::dyn_cast<DoLoopOp>(parent)) {
      popFn(rop);
      return;
    }
    if (auto rop = mlir::dyn_cast<IterWhileOp>(parent)) {
      popFn(rop);
      return;
    }
    // Otherwise, a block argument is provided via the pred blocks.
    for (auto *pred : ba.getOwner()->getPredecessors()) {
      auto u = pred->getTerminator()->getOperand(ba.getArgNumber());
      collectArrayAccessFrom(u);
    }
  }

  // Recursively trace operands to find all array operations relating to the
  // values merged.
  void collectArrayAccessFrom(mlir::Value val) {
    if (!val || visited.contains(val))
      return;
    visited.insert(val);

    // Process a block argument.
    if (auto ba = val.dyn_cast<mlir::BlockArgument>()) {
      collectArrayAccessFrom(ba);
      return;
    }

    // Process an Op.
    if (auto *op = val.getDefiningOp()) {
      collectArrayAccessFrom(op, val);
      return;
    }

    fir::emitFatalError(val.getLoc(), "unhandled value");
  }

  /// Is \op inside the loop nest region ?
  bool opIsInsideLoops(mlir::Operation *op) const {
    return loopRegion && loopRegion->isAncestor(op->getParentRegion());
  }

  /// Recursively trace the use of an operation results, calling
  /// collectArrayAccessFrom on the direct and indirect user operands.
  /// TODO: Replace recursive algorithm on def-use chain with an iterative one
  /// with an explicit stack.
  void followUsers(mlir::Operation *op) {
    for (auto userOperand : op->getOperands())
      collectArrayAccessFrom(userOperand);
    // Go through potential converts/coordinate_op.
    for (mlir::Operation *indirectUser : op->getUsers())
      followUsers(indirectUser);
  }

  llvm::SmallVectorImpl<mlir::Operation *> &reach;
  llvm::SmallPtrSet<mlir::Value, 16> visited;
  /// Region of the loops nest that produced the array value.
  mlir::Region *loopRegion;

public:
  /// Return all ops that produce the array value that is stored into the
  /// `array_merge_store`.
  static void reachingValues(llvm::SmallVectorImpl<mlir::Operation *> &reach,
                             mlir::Value seq) {
    reach.clear();
    mlir::Region *loopRegion = nullptr;
    // Only `DoLoopOp` is tested here since array operations are currently only
    // associated with this kind of loop.
    if (auto doLoop =
            mlir::dyn_cast_or_null<fir::DoLoopOp>(seq.getDefiningOp()))
      loopRegion = &doLoop->getRegion(0);
    ReachCollector collector(reach, loopRegion);
    collector.collectArrayAccessFrom(seq);
  }
};
} // namespace

/// Find all the array operations that access the array value that is loaded by
/// the array load operation, `load`.
const llvm::SmallVector<mlir::Operation *> &
ArrayCopyAnalysis::arrayAccesses(ArrayLoadOp load) {
  auto lmIter = loadMapSets.find(load);
  if (lmIter != loadMapSets.end())
    return lmIter->getSecond();

  llvm::SmallVector<mlir::Operation *> accesses;
  UseSetT visited;
  llvm::SmallVector<mlir::OpOperand *> queue; // uses of ArrayLoad[orig]

  auto appendToQueue = [&](mlir::Value val) {
    for (mlir::OpOperand &use : val.getUses())
      if (!visited.count(&use)) {
        visited.insert(&use);
        queue.push_back(&use);
      }
  };

  // Build the set of uses of `original`.
  // let USES = { uses of original fir.load }
  appendToQueue(load);

  // Process the worklist until done.
  while (!queue.empty()) {
    mlir::OpOperand *operand = queue.pop_back_val();
    mlir::Operation *owner = operand->getOwner();

    auto structuredLoop = [&](auto ro) {
      if (auto blockArg = ro.iterArgToBlockArg(operand->get())) {
        int64_t arg = blockArg.getArgNumber();
        mlir::Value output = ro.getResult(ro.finalValue() ? arg : arg - 1);
        appendToQueue(output);
        appendToQueue(blockArg);
      }
    };
    // TODO: this need to be updated to use the control-flow interface.
    auto branchOp = [&](mlir::Block *dest, OperandRange operands) {
      if (operands.empty())
        return;

      // Check if this operand is within the range.
      unsigned operandIndex = operand->getOperandNumber();
      unsigned operandsStart = operands.getBeginOperandIndex();
      if (operandIndex < operandsStart ||
          operandIndex >= (operandsStart + operands.size()))
        return;

      // Index the successor.
      unsigned argIndex = operandIndex - operandsStart;
      appendToQueue(dest->getArgument(argIndex));
    };
    // Thread uses into structured loop bodies and return value uses.
    if (auto ro = mlir::dyn_cast<DoLoopOp>(owner)) {
      structuredLoop(ro);
    } else if (auto ro = mlir::dyn_cast<IterWhileOp>(owner)) {
      structuredLoop(ro);
    } else if (auto rs = mlir::dyn_cast<ResultOp>(owner)) {
      // Thread any uses of fir.if that return the marked array value.
      if (auto ifOp = rs->getParentOfType<fir::IfOp>())
        appendToQueue(ifOp.getResult(operand->getOperandNumber()));
    } else if (mlir::isa<ArrayFetchOp>(owner)) {
      // Keep track of array value fetches.
      LLVM_DEBUG(llvm::dbgs()
                 << "add fetch {" << *owner << "} to array value set\n");
      accesses.push_back(owner);
    } else if (auto update = mlir::dyn_cast<ArrayUpdateOp>(owner)) {
      // Keep track of array value updates and thread the return value uses.
      LLVM_DEBUG(llvm::dbgs()
                 << "add update {" << *owner << "} to array value set\n");
      accesses.push_back(owner);
      appendToQueue(update.getResult());
    } else if (auto update = mlir::dyn_cast<ArrayModifyOp>(owner)) {
      // Keep track of array value modification and thread the return value
      // uses.
      LLVM_DEBUG(llvm::dbgs()
                 << "add modify {" << *owner << "} to array value set\n");
      accesses.push_back(owner);
      appendToQueue(update.getResult(1));
    } else if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(owner)) {
      branchOp(br.getDest(), br.getDestOperands());
    } else if (auto br = mlir::dyn_cast<mlir::cf::CondBranchOp>(owner)) {
      branchOp(br.getTrueDest(), br.getTrueOperands());
      branchOp(br.getFalseDest(), br.getFalseOperands());
    } else if (mlir::isa<ArrayMergeStoreOp>(owner)) {
      // do nothing
    } else {
      llvm::report_fatal_error("array value reached unexpected op");
    }
  }
  return loadMapSets.insert({load, accesses}).first->getSecond();
}

/// Is there a conflict between the array value that was updated and to be
/// stored to `st` and the set of arrays loaded (`reach`) and used to compute
/// the updated value?
static bool conflictOnLoad(llvm::ArrayRef<mlir::Operation *> reach,
                           ArrayMergeStoreOp st) {
  mlir::Value load;
  mlir::Value addr = st.memref();
  auto stEleTy = fir::dyn_cast_ptrOrBoxEleTy(addr.getType());
  for (auto *op : reach) {
    auto ld = mlir::dyn_cast<ArrayLoadOp>(op);
    if (!ld)
      continue;
    mlir::Type ldTy = ld.memref().getType();
    if (auto boxTy = ldTy.dyn_cast<fir::BoxType>())
      ldTy = boxTy.getEleTy();
    if (ldTy.isa<fir::PointerType>() && stEleTy == dyn_cast_ptrEleTy(ldTy))
      return true;
    if (ld.memref() == addr) {
      if (ld.getResult() != st.original())
        return true;
      if (load)
        return true;
      load = ld;
    }
  }
  return false;
}

/// Check if there is any potential conflict in the chained update operations
/// (ArrayFetchOp, ArrayUpdateOp, ArrayModifyOp) while merging back to the
/// array. A potential conflict is detected if two operations work on the same
/// indices.
static bool conflictOnMerge(llvm::ArrayRef<mlir::Operation *> accesses) {
  if (accesses.size() < 2)
    return false;
  llvm::SmallVector<mlir::Value> indices;
  LLVM_DEBUG(llvm::dbgs() << "check merge conflict on with " << accesses.size()
                          << " accesses on the list\n");
  for (auto *op : accesses) {
    assert((mlir::isa<ArrayFetchOp, ArrayUpdateOp, ArrayModifyOp>(op)) &&
           "unexpected operation in analysis");
    llvm::SmallVector<mlir::Value> compareVector;
    if (auto u = mlir::dyn_cast<ArrayUpdateOp>(op)) {
      if (indices.empty()) {
        indices = u.indices();
        continue;
      }
      compareVector = u.indices();
    } else if (auto f = mlir::dyn_cast<ArrayModifyOp>(op)) {
      if (indices.empty()) {
        indices = f.indices();
        continue;
      }
      compareVector = f.indices();
    } else if (auto f = mlir::dyn_cast<ArrayFetchOp>(op)) {
      if (indices.empty()) {
        indices = f.indices();
        continue;
      }
      compareVector = f.indices();
    }
    if (compareVector != indices)
      return true;
    LLVM_DEBUG(llvm::dbgs() << "vectors compare equal\n");
  }
  return false;
}

// Are either of types of conflicts present?
inline bool conflictDetected(llvm::ArrayRef<mlir::Operation *> reach,
                             llvm::ArrayRef<mlir::Operation *> accesses,
                             ArrayMergeStoreOp st) {
  return conflictOnLoad(reach, st) || conflictOnMerge(accesses);
}

/// Constructor of the array copy analysis.
/// This performs the analysis and saves the intermediate results.
void ArrayCopyAnalysis::construct(mlir::Operation *topLevelOp) {
  topLevelOp->walk([&](Operation *op) {
    if (auto st = mlir::dyn_cast<fir::ArrayMergeStoreOp>(op)) {
      llvm::SmallVector<Operation *> values;
      ReachCollector::reachingValues(values, st.sequence());
      const llvm::SmallVector<Operation *> &accesses =
          arrayAccesses(mlir::cast<ArrayLoadOp>(st.original().getDefiningOp()));
      if (conflictDetected(values, accesses, st)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "CONFLICT: copies required for " << st << '\n'
                   << "   adding conflicts on: " << op << " and "
                   << st.original() << '\n');
        conflicts.insert(op);
        conflicts.insert(st.original().getDefiningOp());
      }
      auto *ld = st.original().getDefiningOp();
      LLVM_DEBUG(llvm::dbgs()
                 << "map: adding {" << *ld << " -> " << st << "}\n");
      useMap.insert({ld, op});
    } else if (auto load = mlir::dyn_cast<ArrayLoadOp>(op)) {
      const llvm::SmallVector<mlir::Operation *> &accesses =
          arrayAccesses(load);
      LLVM_DEBUG(llvm::dbgs() << "process load: " << load
                              << ", accesses: " << accesses.size() << '\n');
      for (auto *acc : accesses) {
        LLVM_DEBUG(llvm::dbgs() << " access: " << *acc << '\n');
        assert((mlir::isa<ArrayFetchOp, ArrayUpdateOp, ArrayModifyOp>(acc)));
        if (!useMap.insert({acc, op}).second) {
          mlir::emitError(
              load.getLoc(),
              "The parallel semantics of multiple array_merge_stores per "
              "array_load are not supported.");
          return;
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "map: adding {" << *acc << "} -> {" << load << "}\n");
      }
    }
  });
}

namespace {
class ArrayLoadConversion : public mlir::OpRewritePattern<ArrayLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayLoadOp load,
                  mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "replace load " << load << " with undef.\n");
    rewriter.replaceOpWithNewOp<UndefOp>(load, load.getType());
    return mlir::success();
  }
};

class ArrayMergeStoreConversion
    : public mlir::OpRewritePattern<ArrayMergeStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayMergeStoreOp store,
                  mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "marking store " << store << " as dead.\n");
    rewriter.eraseOp(store);
    return mlir::success();
  }
};
} // namespace

static mlir::Type getEleTy(mlir::Type ty) {
  if (auto t = dyn_cast_ptrEleTy(ty))
    ty = t;
  if (auto t = ty.dyn_cast<SequenceType>())
    ty = t.getEleTy();
  // FIXME: keep ptr/heap/ref information.
  return ReferenceType::get(ty);
}

// Extract extents from the ShapeOp/ShapeShiftOp into the result vector.
// TODO: getExtents on op should return a ValueRange instead of a vector.
static void getExtents(llvm::SmallVectorImpl<mlir::Value> &result,
                       mlir::Value shape) {
  auto *shapeOp = shape.getDefiningOp();
  if (auto s = mlir::dyn_cast<fir::ShapeOp>(shapeOp)) {
    auto e = s.getExtents();
    result.insert(result.end(), e.begin(), e.end());
    return;
  }
  if (auto s = mlir::dyn_cast<fir::ShapeShiftOp>(shapeOp)) {
    auto e = s.getExtents();
    result.insert(result.end(), e.begin(), e.end());
    return;
  }
  llvm::report_fatal_error("not a fir.shape/fir.shape_shift op");
}

// Place the extents of the array loaded by an ArrayLoadOp into the result
// vector and return a ShapeOp/ShapeShiftOp with the corresponding extents. If
// the ArrayLoadOp is loading a fir.box, code will be generated to read the
// extents from the fir.box, and a the retunred ShapeOp is built with the read
// extents.
// Otherwise, the extents will be extracted from the ShapeOp/ShapeShiftOp
// argument of the ArrayLoadOp that is returned.
static mlir::Value
getOrReadExtentsAndShapeOp(mlir::Location loc, mlir::PatternRewriter &rewriter,
                           fir::ArrayLoadOp loadOp,
                           llvm::SmallVectorImpl<mlir::Value> &result) {
  assert(result.empty());
  if (auto boxTy = loadOp.memref().getType().dyn_cast<fir::BoxType>()) {
    auto rank = fir::dyn_cast_ptrOrBoxEleTy(boxTy)
                    .cast<fir::SequenceType>()
                    .getDimension();
    auto idxTy = rewriter.getIndexType();
    for (decltype(rank) dim = 0; dim < rank; ++dim) {
      auto dimVal = rewriter.create<arith::ConstantIndexOp>(loc, dim);
      auto dimInfo = rewriter.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                                     loadOp.memref(), dimVal);
      result.emplace_back(dimInfo.getResult(1));
    }
    auto shapeType = fir::ShapeType::get(rewriter.getContext(), rank);
    return rewriter.create<fir::ShapeOp>(loc, shapeType, result);
  }
  getExtents(result, loadOp.shape());
  return loadOp.shape();
}

static mlir::Type toRefType(mlir::Type ty) {
  if (fir::isa_ref_type(ty))
    return ty;
  return fir::ReferenceType::get(ty);
}

static mlir::Value
genCoorOp(mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Type eleTy,
          mlir::Type resTy, mlir::Value alloc, mlir::Value shape,
          mlir::Value slice, mlir::ValueRange indices,
          mlir::ValueRange typeparams, bool skipOrig = false) {
  llvm::SmallVector<mlir::Value> originated;
  if (skipOrig)
    originated.assign(indices.begin(), indices.end());
  else
    originated = fir::factory::originateIndices(loc, rewriter, alloc.getType(),
                                                shape, indices);
  auto seqTy = fir::dyn_cast_ptrOrBoxEleTy(alloc.getType());
  assert(seqTy && seqTy.isa<fir::SequenceType>());
  const auto dimension = seqTy.cast<fir::SequenceType>().getDimension();
  mlir::Value result = rewriter.create<fir::ArrayCoorOp>(
      loc, eleTy, alloc, shape, slice,
      llvm::ArrayRef<mlir::Value>{originated}.take_front(dimension),
      typeparams);
  if (dimension < originated.size())
    result = rewriter.create<fir::CoordinateOp>(
        loc, resTy, result,
        llvm::ArrayRef<mlir::Value>{originated}.drop_front(dimension));
  return result;
}

namespace {
/// Conversion of fir.array_update and fir.array_modify Ops.
/// If there is a conflict for the update, then we need to perform a
/// copy-in/copy-out to preserve the original values of the array. If there is
/// no conflict, then it is save to eschew making any copies.
template <typename ArrayOp>
class ArrayUpdateConversionBase : public mlir::OpRewritePattern<ArrayOp> {
public:
  explicit ArrayUpdateConversionBase(mlir::MLIRContext *ctx,
                                     const ArrayCopyAnalysis &a,
                                     const OperationUseMapT &m)
      : mlir::OpRewritePattern<ArrayOp>{ctx}, analysis{a}, useMap{m} {}

  void genArrayCopy(mlir::Location loc, mlir::PatternRewriter &rewriter,
                    mlir::Value dst, mlir::Value src, mlir::Value shapeOp,
                    mlir::Type arrTy) const {
    auto insPt = rewriter.saveInsertionPoint();
    llvm::SmallVector<mlir::Value> indices;
    llvm::SmallVector<mlir::Value> extents;
    getExtents(extents, shapeOp);
    // Build loop nest from column to row.
    for (auto sh : llvm::reverse(extents)) {
      auto idxTy = rewriter.getIndexType();
      auto ubi = rewriter.create<fir::ConvertOp>(loc, idxTy, sh);
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto ub = rewriter.create<arith::SubIOp>(loc, idxTy, ubi, one);
      auto loop = rewriter.create<fir::DoLoopOp>(loc, zero, ub, one);
      rewriter.setInsertionPointToStart(loop.getBody());
      indices.push_back(loop.getInductionVar());
    }
    // Reverse the indices so they are in column-major order.
    std::reverse(indices.begin(), indices.end());
    auto ty = getEleTy(arrTy);
    auto fromAddr = rewriter.create<fir::ArrayCoorOp>(
        loc, ty, src, shapeOp, mlir::Value{},
        fir::factory::originateIndices(loc, rewriter, src.getType(), shapeOp,
                                       indices),
        mlir::ValueRange{});
    auto load = rewriter.create<fir::LoadOp>(loc, fromAddr);
    auto toAddr = rewriter.create<fir::ArrayCoorOp>(
        loc, ty, dst, shapeOp, mlir::Value{},
        fir::factory::originateIndices(loc, rewriter, dst.getType(), shapeOp,
                                       indices),
        mlir::ValueRange{});
    rewriter.create<fir::StoreOp>(loc, load, toAddr);
    rewriter.restoreInsertionPoint(insPt);
  }

  /// Copy the RHS element into the LHS and insert copy-in/copy-out between a
  /// temp and the LHS if the analysis found potential overlaps between the RHS
  /// and LHS arrays. The element copy generator must be provided through \p
  /// assignElement. \p update must be the ArrayUpdateOp or the ArrayModifyOp.
  /// Returns the address of the LHS element inside the loop and the LHS
  /// ArrayLoad result.
  std::pair<mlir::Value, mlir::Value>
  materializeAssignment(mlir::Location loc, mlir::PatternRewriter &rewriter,
                        ArrayOp update,
                        llvm::function_ref<void(mlir::Value)> assignElement,
                        mlir::Type lhsEltRefType) const {
    auto *op = update.getOperation();
    mlir::Operation *loadOp = useMap.lookup(op);
    auto load = mlir::cast<ArrayLoadOp>(loadOp);
    LLVM_DEBUG(llvm::outs() << "does " << load << " have a conflict?\n");
    if (analysis.hasPotentialConflict(loadOp)) {
      // If there is a conflict between the arrays, then we copy the lhs array
      // to a temporary, update the temporary, and copy the temporary back to
      // the lhs array. This yields Fortran's copy-in copy-out array semantics.
      LLVM_DEBUG(llvm::outs() << "Yes, conflict was found\n");
      rewriter.setInsertionPoint(loadOp);
      // Copy in.
      llvm::SmallVector<mlir::Value> extents;
      mlir::Value shapeOp =
          getOrReadExtentsAndShapeOp(loc, rewriter, load, extents);
      auto allocmem = rewriter.create<AllocMemOp>(
          loc, dyn_cast_ptrOrBoxEleTy(load.memref().getType()),
          load.typeparams(), extents);
      genArrayCopy(load.getLoc(), rewriter, allocmem, load.memref(), shapeOp,
                   load.getType());
      rewriter.setInsertionPoint(op);
      mlir::Value coor = genCoorOp(
          rewriter, loc, getEleTy(load.getType()), lhsEltRefType, allocmem,
          shapeOp, load.slice(), update.indices(), load.typeparams(),
          update->hasAttr(fir::factory::attrFortranArrayOffsets()));
      assignElement(coor);
      mlir::Operation *storeOp = useMap.lookup(loadOp);
      auto store = mlir::cast<ArrayMergeStoreOp>(storeOp);
      rewriter.setInsertionPoint(storeOp);
      // Copy out.
      genArrayCopy(store.getLoc(), rewriter, store.memref(), allocmem, shapeOp,
                   load.getType());
      rewriter.create<FreeMemOp>(loc, allocmem);
      return {coor, load.getResult()};
    }
    // Otherwise, when there is no conflict (a possible loop-carried
    // dependence), the lhs array can be updated in place.
    LLVM_DEBUG(llvm::outs() << "No, conflict wasn't found\n");
    rewriter.setInsertionPoint(op);
    auto coorTy = getEleTy(load.getType());
    mlir::Value coor = genCoorOp(
        rewriter, loc, coorTy, lhsEltRefType, load.memref(), load.shape(),
        load.slice(), update.indices(), load.typeparams(),
        update->hasAttr(fir::factory::attrFortranArrayOffsets()));
    assignElement(coor);
    return {coor, load.getResult()};
  }

private:
  const ArrayCopyAnalysis &analysis;
  const OperationUseMapT &useMap;
};

class ArrayUpdateConversion : public ArrayUpdateConversionBase<ArrayUpdateOp> {
public:
  explicit ArrayUpdateConversion(mlir::MLIRContext *ctx,
                                 const ArrayCopyAnalysis &a,
                                 const OperationUseMapT &m)
      : ArrayUpdateConversionBase{ctx, a, m} {}

  mlir::LogicalResult
  matchAndRewrite(ArrayUpdateOp update,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = update.getLoc();
    auto assignElement = [&](mlir::Value coor) {
      rewriter.create<fir::StoreOp>(loc, update.merge(), coor);
    };
    auto lhsEltRefType = toRefType(update.merge().getType());
    auto [_, lhsLoadResult] = materializeAssignment(
        loc, rewriter, update, assignElement, lhsEltRefType);
    update.replaceAllUsesWith(lhsLoadResult);
    rewriter.replaceOp(update, lhsLoadResult);
    return mlir::success();
  }
};

class ArrayModifyConversion : public ArrayUpdateConversionBase<ArrayModifyOp> {
public:
  explicit ArrayModifyConversion(mlir::MLIRContext *ctx,
                                 const ArrayCopyAnalysis &a,
                                 const OperationUseMapT &m)
      : ArrayUpdateConversionBase{ctx, a, m} {}

  mlir::LogicalResult
  matchAndRewrite(ArrayModifyOp modify,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = modify.getLoc();
    auto assignElement = [](mlir::Value) {
      // Assignment already materialized by lowering using lhs element address.
    };
    auto lhsEltRefType = modify.getResult(0).getType();
    auto [lhsEltCoor, lhsLoadResult] = materializeAssignment(
        loc, rewriter, modify, assignElement, lhsEltRefType);
    modify.replaceAllUsesWith(mlir::ValueRange{lhsEltCoor, lhsLoadResult});
    rewriter.replaceOp(modify, mlir::ValueRange{lhsEltCoor, lhsLoadResult});
    return mlir::success();
  }
};

class ArrayFetchConversion : public mlir::OpRewritePattern<ArrayFetchOp> {
public:
  explicit ArrayFetchConversion(mlir::MLIRContext *ctx,
                                const OperationUseMapT &m)
      : OpRewritePattern{ctx}, useMap{m} {}

  mlir::LogicalResult
  matchAndRewrite(ArrayFetchOp fetch,
                  mlir::PatternRewriter &rewriter) const override {
    auto *op = fetch.getOperation();
    rewriter.setInsertionPoint(op);
    auto load = mlir::cast<ArrayLoadOp>(useMap.lookup(op));
    auto loc = fetch.getLoc();
    mlir::Value coor =
        genCoorOp(rewriter, loc, getEleTy(load.getType()),
                  toRefType(fetch.getType()), load.memref(), load.shape(),
                  load.slice(), fetch.indices(), load.typeparams(),
                  fetch->hasAttr(fir::factory::attrFortranArrayOffsets()));
    rewriter.replaceOpWithNewOp<fir::LoadOp>(fetch, coor);
    return mlir::success();
  }

private:
  const OperationUseMapT &useMap;
};
} // namespace

namespace {
class ArrayValueCopyConverter
    : public ArrayValueCopyBase<ArrayValueCopyConverter> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "\n\narray-value-copy pass on function '"
                            << func.getName() << "'\n");
    auto *context = &getContext();

    // Perform the conflict analysis.
    auto &analysis = getAnalysis<ArrayCopyAnalysis>();
    const auto &useMap = analysis.getUseMap();

    // Phase 1 is performing a rewrite on the array accesses. Once all the
    // array accesses are rewritten we can go on phase 2.
    // Phase 2 gets rid of the useless copy-in/copyout operations. The copy-in
    // /copy-out refers the Fortran copy-in/copy-out semantics on statements.
    mlir::RewritePatternSet patterns1(context);
    patterns1.insert<ArrayFetchConversion>(context, useMap);
    patterns1.insert<ArrayUpdateConversion>(context, analysis, useMap);
    patterns1.insert<ArrayModifyConversion>(context, analysis, useMap);
    mlir::ConversionTarget target(*context);
    target.addLegalDialect<
        FIROpsDialect, mlir::scf::SCFDialect, mlir::arith::ArithmeticDialect,
        mlir::cf::ControlFlowDialect, mlir::StandardOpsDialect>();
    target.addIllegalOp<ArrayFetchOp, ArrayUpdateOp, ArrayModifyOp>();
    // Rewrite the array fetch and array update ops.
    if (mlir::failed(
            mlir::applyPartialConversion(func, target, std::move(patterns1)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in array-value-copy pass, phase 1");
      signalPassFailure();
    }

    mlir::RewritePatternSet patterns2(context);
    patterns2.insert<ArrayLoadConversion>(context);
    patterns2.insert<ArrayMergeStoreConversion>(context);
    target.addIllegalOp<ArrayLoadOp, ArrayMergeStoreOp>();
    if (mlir::failed(
            mlir::applyPartialConversion(func, target, std::move(patterns2)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in array-value-copy pass, phase 2");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> fir::createArrayValueCopyPass() {
  return std::make_unique<ArrayValueCopyConverter>();
}
