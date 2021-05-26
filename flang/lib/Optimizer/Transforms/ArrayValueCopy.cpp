//===-- ArrayValueCopy.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Lower/Todo.h" // delete!
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Transforms/Factory.h"
#include "flang/Optimizer/Transforms/Passes.h"
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
  using LoadMapSetsT = llvm::DenseMap<mlir::Operation *, UseSetT>;

  ArrayCopyAnalysis(mlir::Operation *op) : operation{op} {
    construct(op->getRegions());
  }

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

  /// For ArrayLoad `load`, return the transitive set of all OpOperands.
  UseSetT getLoadUseSet(mlir::Operation *load) const {
    assert(loadMapSets.count(load) && "analysis missed an array load?");
    return loadMapSets.lookup(load);
  }

  /// Get all the array value operations that use the original array value
  /// as passed to `store`.
  void arrayAccesses(llvm::SmallVectorImpl<mlir::Operation *> &accesses,
                     ArrayLoadOp load);

private:
  void construct(mlir::MutableArrayRef<mlir::Region> regions);

  mlir::Operation *operation; // operation that analysis ran upon
  ConflictSetT conflicts;     // set of conflicts (loads and merge stores)
  OperationUseMapT useMap;
  LoadMapSetsT loadMapSets;
};
} // namespace

// Recursively trace operands to find all array operations relating to the
// values merged.
static void populateSets(llvm::SmallVectorImpl<mlir::Operation *> &reach,
                         llvm::SmallPtrSetImpl<mlir::Value> &visited,
                         mlir::Value val) {
  if (!val || visited.contains(val))
    return;
  visited.insert(val);

  if (auto *op = val.getDefiningOp()) {
    // `val` is defined by an Op, process the defining Op.
    // If `val` is defined by a region containing Op, we want to drill down and
    // through that Op's region(s).
    LLVM_DEBUG(llvm::dbgs() << "popset: " << val << '\n');
    auto popFn = [&](auto rop) {
      auto resNum = val.cast<mlir::OpResult>().getResultNumber();
      llvm::SmallVector<mlir::Value> results;
      rop.resultToSourceOps(results, resNum);
      for (auto u : results)
        populateSets(reach, visited, u);
    };
    if (auto rop = mlir::dyn_cast<DoLoopOp>(op)) {
      popFn(rop);
      return;
    }
    if (auto rop = mlir::dyn_cast<IterWhileOp>(op)) {
      popFn(rop);
      return;
    }
    if (auto rop = mlir::dyn_cast<fir::IfOp>(op)) {
      popFn(rop);
      return;
    }

    // Otherwise, Op does not contain a region so just chase its operands.
    if (mlir::isa<ArrayLoadOp, ArrayUpdateOp, ArrayFetchOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "add " << *op << " to reachable set\n");
      reach.emplace_back(op);
    }
    for (auto u : op->getOperands())
      populateSets(reach, visited, u);
    return;
  }

  // Process a block argument.
  auto ba = val.cast<mlir::BlockArgument>();
  auto *parent = ba.getOwner()->getParentOp();
  // If inside an Op holding a region, the block argument corresponds to an
  // argument passed to the containing Op.
  auto popFn = [&](auto rop) {
    populateSets(reach, visited, rop.blockArgToSourceOp(ba.getArgNumber()));
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
  for (auto pred : ba.getOwner()->getPredecessors()) {
    auto u = pred->getTerminator()->getOperand(ba.getArgNumber());
    populateSets(reach, visited, u);
  }
}

/// Return all ops that produce the array value that is stored into the
/// `array_merge_store`.
static void reachingValues(llvm::SmallVectorImpl<mlir::Operation *> &reach,
                           mlir::Value seq) {
  reach.clear();
  llvm::SmallPtrSet<mlir::Value, 16> visited;
  populateSets(reach, visited, seq);
}

/// Find all the array operations that access the array value that is loaded by
/// the array load operation, `load`.
void ArrayCopyAnalysis::arrayAccesses(
    llvm::SmallVectorImpl<mlir::Operation *> &accesses, ArrayLoadOp load) {
  accesses.clear();
  auto lmIter = loadMapSets.find(load);
  if (lmIter != loadMapSets.end()) {
    for (auto *opnd : lmIter->second) {
      auto *owner = opnd->getOwner();
      if (mlir::isa<ArrayFetchOp, ArrayUpdateOp>(owner))
        accesses.push_back(owner);
    }
    return;
  }

  UseSetT visited;
  llvm::SmallVector<mlir::OpOperand *> queue; // uses of ArrayLoad[orig]

  auto appendToQueue = [&](mlir::Value val) {
    for (auto &use : val.getUses())
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
    auto *operand = queue.pop_back_val();
    auto *owner = operand->getOwner();
    if (!owner)
      continue;
    auto structuredLoop = [&](auto ro) {
      if (auto blockArg = ro.iterArgToBlockArg(operand->get())) {
        auto arg = blockArg.getArgNumber();
        auto output = ro.getResult(ro.finalValue() ? arg : arg - 1);
        appendToQueue(output);
        appendToQueue(blockArg);
      }
    };
    auto branchOp = [&](mlir::Block *dest, auto operands) {
      for (auto i : llvm::enumerate(operands))
        if (operand->get() == i.value()) {
          auto blockArg = dest->getArgument(i.index());
          appendToQueue(blockArg);
        }
    };
    // Thread uses into structured loop bodies and return value uses.
    if (auto ro = mlir::dyn_cast<DoLoopOp>(owner)) {
      structuredLoop(ro);
    } else if (auto ro = mlir::dyn_cast<IterWhileOp>(owner)) {
      structuredLoop(ro);
    } else if (auto rs = mlir::dyn_cast<ResultOp>(owner)) {
      // Thread any uses of fir.if that return the marked array value.
      auto *parent = rs->getParentRegion()->getParentOp();
      if (auto ifOp = mlir::dyn_cast<fir::IfOp>(parent))
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
    } else if (auto br = mlir::dyn_cast<mlir::BranchOp>(owner)) {
      branchOp(br.getDest(), br.destOperands());
    } else if (auto br = mlir::dyn_cast<mlir::CondBranchOp>(owner)) {
      branchOp(br.getTrueDest(), br.getTrueOperands());
      branchOp(br.getFalseDest(), br.getFalseOperands());
    } else if (mlir::isa<ArrayMergeStoreOp>(owner)) {
      // do nothing
    } else {
      llvm::report_fatal_error("array value reached unexpected op");
    }
  }
  loadMapSets.insert({load, visited});
}

/// Is there a conflict between the array value that was updated and to be
/// stored to `st` and the set of arrays loaded (`reach`) and used to compute
/// the updated value?
static bool conflictOnLoad(llvm::ArrayRef<mlir::Operation *> reach,
                           ArrayMergeStoreOp st) {
  mlir::Value load;
  auto addr = st.memref();
  auto stEleTy = fir::dyn_cast_ptrOrBoxEleTy(addr.getType());
  for (auto *op : reach)
    if (auto ld = mlir::dyn_cast<ArrayLoadOp>(op)) {
      auto ldTy = ld.memref().getType();
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

static bool conflictOnMerge(llvm::ArrayRef<mlir::Operation *> accesses) {
  if (accesses.size() < 2)
    return false;
  llvm::SmallVector<mlir::Value> indices;
  LLVM_DEBUG(llvm::dbgs() << "check merge conflict on with " << accesses.size()
                          << " accesses on the list\n");
  for (auto *op : accesses) {
    llvm::SmallVector<mlir::Value> compareVector;
    if (auto u = mlir::dyn_cast<ArrayUpdateOp>(op)) {
      if (indices.empty()) {
        indices = u.indices();
        continue;
      }
      compareVector = u.indices();
    } else if (auto f = mlir::dyn_cast<ArrayFetchOp>(op)) {
      if (indices.empty()) {
        indices = f.indices();
        continue;
      }
      compareVector = f.indices();
    } else {
      mlir::emitError(op->getLoc(), "unexpected operation in analysis");
    }
    if (compareVector.size() != indices.size() ||
        llvm::any_of(llvm::zip(compareVector, indices), [&](auto pair) {
          return std::get<0>(pair) != std::get<1>(pair);
        }))
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
void ArrayCopyAnalysis::construct(mlir::MutableArrayRef<mlir::Region> regions) {
  for (auto &region : regions)
    for (auto &block : region.getBlocks())
      for (auto &op : block.getOperations()) {
        if (op.getNumRegions())
          construct(op.getRegions());
        if (auto st = mlir::dyn_cast<fir::ArrayMergeStoreOp>(op)) {
          llvm::SmallVector<Operation *> values;
          reachingValues(values, st.sequence());
          llvm::SmallVector<Operation *> accesses;
          arrayAccesses(accesses,
                        mlir::cast<ArrayLoadOp>(st.original().getDefiningOp()));
          if (conflictDetected(values, accesses, st)) {
            LLVM_DEBUG(llvm::dbgs()
                       << "CONFLICT: copies required for " << st << '\n'
                       << "   adding conflicts on: " << op << " and "
                       << st.original() << '\n');
            conflicts.insert(&op);
            conflicts.insert(st.original().getDefiningOp());
          }
          auto *ld = st.original().getDefiningOp();
          LLVM_DEBUG(llvm::dbgs()
                     << "map: adding {" << *ld << " -> " << st << "}\n");
          useMap.insert({ld, &op});
        } else if (auto load = mlir::dyn_cast<ArrayLoadOp>(op)) {
          llvm::SmallVector<mlir::Operation *> accesses;
          arrayAccesses(accesses, load);
          LLVM_DEBUG(llvm::dbgs() << "process load: " << load
                                  << ", accesses: " << accesses.size() << '\n');
          for (auto *acc : accesses) {
            LLVM_DEBUG(llvm::dbgs() << " access: " << *acc << '\n');
            if (mlir::isa<ArrayFetchOp, ArrayUpdateOp>(acc)) {
              if (useMap.count(acc)) {
                mlir::emitError(
                    load.getLoc(),
                    "The parallel semantics of multiple array_merge_stores per "
                    "array_load are not supported.");
                continue;
              }
              LLVM_DEBUG(llvm::dbgs() << "map: adding {" << *acc << "} -> {"
                                      << load << "}\n");
              useMap.insert({acc, &op});
            }
          }
        }
      }
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

/// Get origins from fir.shape_shit/fir.shift op. Empty result if
/// \p shapeVal is empty or is a fir.shape.
static std::vector<mlir::Value> getOrigins(mlir::Value shapeVal) {
  if (shapeVal)
    if (auto *shapeOp = shapeVal.getDefiningOp()) {
      if (auto shOp = mlir::dyn_cast<fir::ShapeShiftOp>(shapeOp))
        return shOp.getOrigins();
      else if (auto shOp = mlir::dyn_cast<fir::ShiftOp>(shapeOp))
        return shOp.getOrigins();
    }
  return {};
}

/// Convert the normalized indices on array_fetch and array_update to the
/// dynamic (and non-zero) origin required by array_coor.
static std::vector<mlir::Value>
originateIndices(mlir::Location loc, mlir::PatternRewriter &rewriter,
                 mlir::Value shapeVal, mlir::ValueRange indices) {
  std::vector<mlir::Value> result;
  auto origins = getOrigins(shapeVal);
  if (!origins.empty()) {
    assert(indices.size() == origins.size());
    for (auto [i, lb] : llvm::zip(indices, origins))
      result.push_back(rewriter.create<mlir::AddIOp>(loc, i, lb));
    return result;
  }
  assert(!shapeVal || mlir::isa<fir::ShapeOp>(shapeVal.getDefiningOp()));
  auto one = rewriter.create<mlir::ConstantIndexOp>(loc, 1);
  for (auto v : indices)
    result.push_back(rewriter.create<mlir::AddIOp>(loc, one, v));
  return result;
}

// Extract extents from the ShapeOp/ShapeShiftOp into the result vector.
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
      auto dimVal = rewriter.create<mlir::ConstantIndexOp>(loc, dim);
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

namespace {
/// Conversion of fir.array_update Op.
/// If there is a conflict for the update, then we need to perform a
/// copy-in/copy-out to preserve the original values of the array. If there is
/// no conflict, then it is save to eschew making any copies.
class ArrayUpdateConversion : public mlir::OpRewritePattern<ArrayUpdateOp> {
public:
  explicit ArrayUpdateConversion(mlir::MLIRContext *ctx,
                                 const ArrayCopyAnalysis &a,
                                 const OperationUseMapT &m)
      : OpRewritePattern{ctx}, analysis{a}, useMap{m} {}

  mlir::LogicalResult
  matchAndRewrite(ArrayUpdateOp update,
                  mlir::PatternRewriter &rewriter) const override {
    auto *op = update.getOperation();
    auto *loadOp = useMap.lookup(op);
    auto load = mlir::cast<ArrayLoadOp>(loadOp);
    LLVM_DEBUG(llvm::outs() << "does " << load << " have a conflict?\n");
    auto loc = update.getLoc();
    auto copyElement = [&](mlir::Value coor) {
      auto input = update.merge();
      if (auto inEleTy = fir::dyn_cast_ptrEleTy(input.getType())) {
        auto outEleTy = fir::unwrapSequenceType(update.getType());
        if (auto inChrTy = inEleTy.dyn_cast<fir::CharacterType>()) {
          assert(outEleTy.isa<fir::CharacterType>());
          fir::factory::genCharacterCopy(input, recoverCharLen(input), coor,
                                         recoverCharLen(coor), rewriter, loc);
        } else if (inEleTy.isa<fir::RecordType>()) {
          TODO(loc, "copy derived type");
        } else {
          llvm::report_fatal_error("not a legal reference type");
        }
      } else {
        rewriter.create<fir::StoreOp>(loc, input, coor);
      }
    };

    if (analysis.hasPotentialConflict(loadOp)) {
      // If there is a conflict between the arrays, then we copy the lhs array
      // to a temporary, update the temporary, and copy the temporary back to
      // the lhs array. This yields Fortran's copy-in copy-out array semantics.
      LLVM_DEBUG(llvm::outs() << "Yes, conflict was found\n");
      rewriter.setInsertionPoint(loadOp);
      // Copy in.
      llvm::SmallVector<mlir::Value> extents;
      auto shapeOp = getOrReadExtentsAndShapeOp(loc, rewriter, load, extents);
      auto allocmem = rewriter.create<AllocMemOp>(
          loc, dyn_cast_ptrOrBoxEleTy(load.memref().getType()),
          load.typeparams(), extents);
      genArrayCopy(load.getLoc(), rewriter, allocmem, load.memref(), shapeOp,
                   load.getType());
      rewriter.setInsertionPoint(op);
      auto coor = rewriter.create<ArrayCoorOp>(
          loc, getEleTy(load.getType()), allocmem, shapeOp, load.slice(),
          originateIndices(loc, rewriter, shapeOp, update.indices()),
          load.typeparams());
      copyElement(coor);
      auto *storeOp = useMap.lookup(loadOp);
      rewriter.setInsertionPoint(storeOp);
      // Copy out.
      auto store = mlir::cast<ArrayMergeStoreOp>(storeOp);
      genArrayCopy(store.getLoc(), rewriter, store.memref(), allocmem, shapeOp,
                   load.getType());
      rewriter.create<FreeMemOp>(loc, allocmem);
    } else {
      // Otherwise, when there is no conflict (a possible loop-carried
      // dependence), the lhs array can be updated in place.
      LLVM_DEBUG(llvm::outs() << "No, conflict wasn't found\n");
      rewriter.setInsertionPoint(op);
      auto coorTy = getEleTy(load.getType());
      auto coor = rewriter.create<ArrayCoorOp>(
          loc, coorTy, load.memref(), load.shape(), load.slice(),
          originateIndices(loc, rewriter, load.shape(), update.indices()),
          load.typeparams());
      copyElement(coor);
    }
    update.replaceAllUsesWith(load.getResult());
    rewriter.replaceOp(update, load.getResult());
    return mlir::success();
  }

  static llvm::SmallVector<mlir::Value> recoverTypeParams(mlir::Value val) {
    auto *op = val.getDefiningOp();
    if (!fir::hasDynamicSize(fir::dyn_cast_ptrEleTy(val.getType())))
      return {};
    if (auto co = mlir::dyn_cast<fir::ConvertOp>(op))
      return recoverTypeParams(co.value());
    if (auto ao = mlir::dyn_cast<fir::ArrayFetchOp>(op))
      return {ao.typeparams().begin(), ao.typeparams().end()};
    if (auto ao = mlir::dyn_cast<fir::ArrayUpdateOp>(op))
      return {ao.typeparams().begin(), ao.typeparams().end()};
    if (auto ao = mlir::dyn_cast<fir::ArrayLoadOp>(op))
      return {ao.typeparams().begin(), ao.typeparams().end()};
    if (auto ao = mlir::dyn_cast<fir::ArrayCoorOp>(op))
      return {ao.typeparams().begin(), ao.typeparams().end()};
    if (auto ao = mlir::dyn_cast<fir::AllocaOp>(op))
      return {ao.typeparams().begin(), ao.typeparams().end()};
    if (auto ao = mlir::dyn_cast<fir::AllocMemOp>(op))
      return {ao.typeparams().begin(), ao.typeparams().end()};
    llvm::report_fatal_error("unexpected buffer");
  }

  static mlir::Value recoverCharLen(mlir::Value val) {
    auto params = recoverTypeParams(val);
    return params.empty() ? mlir::Value{} : params[0];
  }

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
      auto zero = rewriter.create<mlir::ConstantIndexOp>(loc, 0);
      auto one = rewriter.create<mlir::ConstantIndexOp>(loc, 1);
      auto ub = rewriter.create<mlir::SubIOp>(loc, idxTy, ubi, one);
      auto loop = rewriter.create<fir::DoLoopOp>(loc, zero, ub, one);
      rewriter.setInsertionPointToStart(loop.getBody());
      indices.push_back(loop.getInductionVar());
    }
    // Reverse the indices so they are in column-major order.
    std::reverse(indices.begin(), indices.end());
    auto ty = getEleTy(arrTy);
    auto fromAddr = rewriter.create<fir::ArrayCoorOp>(
        loc, ty, src, shapeOp, mlir::Value{},
        originateIndices(loc, rewriter, shapeOp, indices), mlir::ValueRange{});
    auto load = rewriter.create<fir::LoadOp>(loc, fromAddr);
    auto toAddr = rewriter.create<fir::ArrayCoorOp>(
        loc, ty, dst, shapeOp, mlir::Value{},
        originateIndices(loc, rewriter, shapeOp, indices), mlir::ValueRange{});
    rewriter.create<fir::StoreOp>(loc, load, toAddr);
    rewriter.restoreInsertionPoint(insPt);
  }

private:
  const ArrayCopyAnalysis &analysis;
  const OperationUseMapT &useMap;
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
    auto coor = rewriter.create<ArrayCoorOp>(
        loc, getEleTy(load.getType()), load.memref(), load.shape(),
        load.slice(),
        originateIndices(loc, rewriter, load.shape(), fetch.indices()),
        load.typeparams());
    if (fir::isa_ref_type(fetch.getType()))
      rewriter.replaceOp(fetch, coor.getResult());
    else
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
  void runOnFunction() override {
    auto func = getFunction();
    LLVM_DEBUG(llvm::dbgs() << "\n\narray-value-copy pass on function '"
                            << func.getName() << "'\n");
    auto *context = &getContext();

    // Perform the conflict analysis.
    auto &analysis = getAnalysis<ArrayCopyAnalysis>();
    const auto &useMap = analysis.getUseMap();

    mlir::OwningRewritePatternList patterns1(context);
    patterns1.insert<ArrayFetchConversion>(context, useMap);
    patterns1.insert<ArrayUpdateConversion>(context, analysis, useMap);
    mlir::ConversionTarget target(*context);
    target.addLegalDialect<FIROpsDialect, mlir::scf::SCFDialect,
                           mlir::StandardOpsDialect>();
    target.addIllegalOp<ArrayFetchOp, ArrayUpdateOp>();
    // Rewrite the array fetch and array update ops.
    if (mlir::failed(
            mlir::applyPartialConversion(func, target, std::move(patterns1)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in array-value-copy pass, phase 1");
      signalPassFailure();
    }

    mlir::OwningRewritePatternList patterns2(context);
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
