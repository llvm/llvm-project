//===-- ArrayValueCopy.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Array.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Factory.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-array-value-copy"

using namespace fir;
using namespace mlir;

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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ArrayCopyAnalysis)

  using ConflictSetT = llvm::SmallPtrSet<mlir::Operation *, 16>;
  using UseSetT = llvm::SmallPtrSet<mlir::OpOperand *, 8>;
  using LoadMapSetsT = llvm::DenseMap<mlir::Operation *, UseSetT>;
  using AmendAccessSetT = llvm::SmallPtrSet<mlir::Operation *, 4>;

  ArrayCopyAnalysis(mlir::Operation *op) : operation{op} { construct(op); }

  mlir::Operation *getOperation() const { return operation; }

  /// Return true iff the `array_merge_store` has potential conflicts.
  bool hasPotentialConflict(mlir::Operation *op) const {
    LLVM_DEBUG(llvm::dbgs()
               << "looking for a conflict on " << *op
               << " and the set has a total of " << conflicts.size() << '\n');
    return conflicts.contains(op);
  }

  /// Return the use map.
  /// The use map maps array access, amend, fetch and update operations back to
  /// the array load that is the original source of the array value.
  /// It maps an array_load to an array_merge_store, if and only if the loaded
  /// array value has pending modifications to be merged.
  const OperationUseMapT &getUseMap() const { return useMap; }

  /// Return the set of array_access ops directly associated with array_amend
  /// ops.
  bool inAmendAccessSet(mlir::Operation *op) const {
    return amendAccesses.count(op);
  }

  /// For ArrayLoad `load`, return the transitive set of all OpOperands.
  UseSetT getLoadUseSet(mlir::Operation *load) const {
    assert(loadMapSets.count(load) && "analysis missed an array load?");
    return loadMapSets.lookup(load);
  }

  void arrayMentions(llvm::SmallVectorImpl<mlir::Operation *> &mentions,
                     ArrayLoadOp load);

private:
  void construct(mlir::Operation *topLevelOp);

  mlir::Operation *operation; // operation that analysis ran upon
  ConflictSetT conflicts;     // set of conflicts (loads and merge stores)
  OperationUseMapT useMap;
  LoadMapSetsT loadMapSets;
  // Set of array_access ops associated with array_amend ops.
  AmendAccessSetT amendAccesses;
};
} // namespace

namespace {
/// Helper class to collect all array operations that produced an array value.
class ReachCollector {
public:
  ReachCollector(llvm::SmallVectorImpl<mlir::Operation *> &reach,
                 mlir::Region *loopRegion)
      : reach{reach}, loopRegion{loopRegion} {}

  void collectArrayMentionFrom(mlir::Operation *op, mlir::ValueRange range) {
    if (range.empty()) {
      collectArrayMentionFrom(op, mlir::Value{});
      return;
    }
    for (mlir::Value v : range)
      collectArrayMentionFrom(v);
  }

  // Collect all the array_access ops in `block`. This recursively looks into
  // blocks in ops with regions.
  // FIXME: This is temporarily relying on the array_amend appearing in a
  // do_loop Region.  This phase ordering assumption can be eliminated by using
  // dominance information to find the array_access ops or by scanning the
  // transitive closure of the amending array_access's users and the defs that
  // reach them.
  void collectAccesses(llvm::SmallVector<ArrayAccessOp> &result,
                       mlir::Block *block) {
    for (auto &op : *block) {
      if (auto access = mlir::dyn_cast<ArrayAccessOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "adding access: " << access << '\n');
        result.push_back(access);
        continue;
      }
      for (auto &region : op.getRegions())
        for (auto &bb : region.getBlocks())
          collectAccesses(result, &bb);
    }
  }

  void collectArrayMentionFrom(mlir::Operation *op, mlir::Value val) {
    // `val` is defined by an Op, process the defining Op.
    // If `val` is defined by a region containing Op, we want to drill down
    // and through that Op's region(s).
    LLVM_DEBUG(llvm::dbgs() << "popset: " << *op << '\n');
    auto popFn = [&](auto rop) {
      assert(val && "op must have a result value");
      auto resNum = val.cast<mlir::OpResult>().getResultNumber();
      llvm::SmallVector<mlir::Value> results;
      rop.resultToSourceOps(results, resNum);
      for (auto u : results)
        collectArrayMentionFrom(u);
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
    if (auto box = mlir::dyn_cast<EmboxOp>(op)) {
      for (auto *user : box.getMemref().getUsers())
        if (user != op)
          collectArrayMentionFrom(user, user->getResults());
      return;
    }
    if (auto mergeStore = mlir::dyn_cast<ArrayMergeStoreOp>(op)) {
      if (opIsInsideLoops(mergeStore))
        collectArrayMentionFrom(mergeStore.getSequence());
      return;
    }

    if (mlir::isa<AllocaOp, AllocMemOp>(op)) {
      // Look for any stores inside the loops, and collect an array operation
      // that produced the value being stored to it.
      for (auto *user : op->getUsers())
        if (auto store = mlir::dyn_cast<fir::StoreOp>(user))
          if (opIsInsideLoops(store))
            collectArrayMentionFrom(store.getValue());
      return;
    }

    // Scan the uses of amend's memref
    if (auto amend = mlir::dyn_cast<ArrayAmendOp>(op)) {
      reach.push_back(op);
      llvm::SmallVector<ArrayAccessOp> accesses;
      collectAccesses(accesses, op->getBlock());
      for (auto access : accesses)
        collectArrayMentionFrom(access.getResult());
    }

    // Otherwise, Op does not contain a region so just chase its operands.
    if (mlir::isa<ArrayAccessOp, ArrayLoadOp, ArrayUpdateOp, ArrayModifyOp,
                  ArrayFetchOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "add " << *op << " to reachable set\n");
      reach.push_back(op);
    }

    // Include all array_access ops using an array_load.
    if (auto arrLd = mlir::dyn_cast<ArrayLoadOp>(op))
      for (auto *user : arrLd.getResult().getUsers())
        if (mlir::isa<ArrayAccessOp>(user)) {
          LLVM_DEBUG(llvm::dbgs() << "add " << *user << " to reachable set\n");
          reach.push_back(user);
        }

    // Array modify assignment is performed on the result. So the analysis must
    // look at the what is done with the result.
    if (mlir::isa<ArrayModifyOp>(op))
      for (auto *user : op->getResult(0).getUsers())
        followUsers(user);

    if (mlir::isa<fir::CallOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "add " << *op << " to reachable set\n");
      reach.push_back(op);
    }

    for (auto u : op->getOperands())
      collectArrayMentionFrom(u);
  }

  void collectArrayMentionFrom(mlir::BlockArgument ba) {
    auto *parent = ba.getOwner()->getParentOp();
    // If inside an Op holding a region, the block argument corresponds to an
    // argument passed to the containing Op.
    auto popFn = [&](auto rop) {
      collectArrayMentionFrom(rop.blockArgToSourceOp(ba.getArgNumber()));
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
      collectArrayMentionFrom(u);
    }
  }

  // Recursively trace operands to find all array operations relating to the
  // values merged.
  void collectArrayMentionFrom(mlir::Value val) {
    if (!val || visited.contains(val))
      return;
    visited.insert(val);

    // Process a block argument.
    if (auto ba = val.dyn_cast<mlir::BlockArgument>()) {
      collectArrayMentionFrom(ba);
      return;
    }

    // Process an Op.
    if (auto *op = val.getDefiningOp()) {
      collectArrayMentionFrom(op, val);
      return;
    }

    emitFatalError(val.getLoc(), "unhandled value");
  }

  /// Return all ops that produce the array value that is stored into the
  /// `array_merge_store`.
  static void reachingValues(llvm::SmallVectorImpl<mlir::Operation *> &reach,
                             mlir::Value seq) {
    reach.clear();
    mlir::Region *loopRegion = nullptr;
    if (auto doLoop = mlir::dyn_cast_or_null<DoLoopOp>(seq.getDefiningOp()))
      loopRegion = &doLoop->getRegion(0);
    ReachCollector collector(reach, loopRegion);
    collector.collectArrayMentionFrom(seq);
  }

private:
  /// Is \op inside the loop nest region ?
  /// FIXME: replace this structural dependence with graph properties.
  bool opIsInsideLoops(mlir::Operation *op) const {
    auto *region = op->getParentRegion();
    while (region) {
      if (region == loopRegion)
        return true;
      region = region->getParentRegion();
    }
    return false;
  }

  /// Recursively trace the use of an operation results, calling
  /// collectArrayMentionFrom on the direct and indirect user operands.
  void followUsers(mlir::Operation *op) {
    for (auto userOperand : op->getOperands())
      collectArrayMentionFrom(userOperand);
    // Go through potential converts/coordinate_op.
    for (auto indirectUser : op->getUsers())
      followUsers(indirectUser);
  }

  llvm::SmallVectorImpl<mlir::Operation *> &reach;
  llvm::SmallPtrSet<mlir::Value, 16> visited;
  /// Region of the loops nest that produced the array value.
  mlir::Region *loopRegion;
};
} // namespace

/// Find all the array operations that access the array value that is loaded by
/// the array load operation, `load`.
void ArrayCopyAnalysis::arrayMentions(
    llvm::SmallVectorImpl<mlir::Operation *> &mentions, ArrayLoadOp load) {
  mentions.clear();
  auto lmIter = loadMapSets.find(load);
  if (lmIter != loadMapSets.end()) {
    for (auto *opnd : lmIter->second) {
      auto *owner = opnd->getOwner();
      if (mlir::isa<ArrayAccessOp, ArrayAmendOp, ArrayFetchOp, ArrayUpdateOp,
                    ArrayModifyOp>(owner))
        mentions.push_back(owner);
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
    mlir::OpOperand *operand = queue.pop_back_val();
    mlir::Operation *owner = operand->getOwner();
    if (!owner)
      continue;
    auto structuredLoop = [&](auto ro) {
      if (auto blockArg = ro.iterArgToBlockArg(operand->get())) {
        int64_t arg = blockArg.getArgNumber();
        mlir::Value output = ro.getResult(ro.getFinalValue() ? arg : arg - 1);
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
      mlir::Operation *parent = rs->getParentRegion()->getParentOp();
      if (auto ifOp = mlir::dyn_cast<fir::IfOp>(parent))
        appendToQueue(ifOp.getResult(operand->getOperandNumber()));
    } else if (mlir::isa<ArrayFetchOp>(owner)) {
      // Keep track of array value fetches.
      LLVM_DEBUG(llvm::dbgs()
                 << "add fetch {" << *owner << "} to array value set\n");
      mentions.push_back(owner);
    } else if (auto update = mlir::dyn_cast<ArrayUpdateOp>(owner)) {
      // Keep track of array value updates and thread the return value uses.
      LLVM_DEBUG(llvm::dbgs()
                 << "add update {" << *owner << "} to array value set\n");
      mentions.push_back(owner);
      appendToQueue(update.getResult());
    } else if (auto update = mlir::dyn_cast<ArrayModifyOp>(owner)) {
      // Keep track of array value modification and thread the return value
      // uses.
      LLVM_DEBUG(llvm::dbgs()
                 << "add modify {" << *owner << "} to array value set\n");
      mentions.push_back(owner);
      appendToQueue(update.getResult(1));
    } else if (auto mention = mlir::dyn_cast<ArrayAccessOp>(owner)) {
      mentions.push_back(owner);
    } else if (auto amend = mlir::dyn_cast<ArrayAmendOp>(owner)) {
      mentions.push_back(owner);
      appendToQueue(amend.getResult());
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
  loadMapSets.insert({load, visited});
}

static bool hasPointerType(mlir::Type type) {
  if (auto boxTy = type.dyn_cast<BoxType>())
    type = boxTy.getEleTy();
  return type.isa<fir::PointerType>();
}

// This is a NF performance hack. It makes a simple test that the slices of the
// load, \p ld, and the merge store, \p st, are trivially mutually exclusive.
static bool mutuallyExclusiveSliceRange(ArrayLoadOp ld, ArrayMergeStoreOp st) {
  // If the same array_load, then no further testing is warranted.
  if (ld.getResult() == st.getOriginal())
    return false;

  auto getSliceOp = [](mlir::Value val) -> SliceOp {
    if (!val)
      return {};
    auto sliceOp = mlir::dyn_cast_or_null<SliceOp>(val.getDefiningOp());
    if (!sliceOp)
      return {};
    return sliceOp;
  };

  auto ldSlice = getSliceOp(ld.getSlice());
  auto stSlice = getSliceOp(st.getSlice());
  if (!ldSlice || !stSlice)
    return false;

  // Resign on subobject slices.
  if (!ldSlice.getFields().empty() || !stSlice.getFields().empty() ||
      !ldSlice.getSubstr().empty() || !stSlice.getSubstr().empty())
    return false;

  // Crudely test that the two slices do not overlap by looking for the
  // following general condition. If the slices look like (i:j) and (j+1:k) then
  // these ranges do not overlap. The addend must be a constant.
  auto ldTriples = ldSlice.getTriples();
  auto stTriples = stSlice.getTriples();
  const auto size = ldTriples.size();
  if (size != stTriples.size())
    return false;

  auto displacedByConstant = [](mlir::Value v1, mlir::Value v2) {
    auto removeConvert = [](mlir::Value v) -> mlir::Operation * {
      auto *op = v.getDefiningOp();
      while (auto conv = mlir::dyn_cast_or_null<ConvertOp>(op))
        op = conv.getValue().getDefiningOp();
      return op;
    };

    auto isPositiveConstant = [](mlir::Value v) -> bool {
      if (auto conOp =
              mlir::dyn_cast<mlir::arith::ConstantOp>(v.getDefiningOp()))
        if (auto iattr = conOp.getValue().dyn_cast<mlir::IntegerAttr>())
          return iattr.getInt() > 0;
      return false;
    };

    auto *op1 = removeConvert(v1);
    auto *op2 = removeConvert(v2);
    if (!op1 || !op2)
      return false;
    if (auto addi = mlir::dyn_cast<mlir::arith::AddIOp>(op2))
      if ((addi.getLhs().getDefiningOp() == op1 &&
           isPositiveConstant(addi.getRhs())) ||
          (addi.getRhs().getDefiningOp() == op1 &&
           isPositiveConstant(addi.getLhs())))
        return true;
    if (auto subi = mlir::dyn_cast<mlir::arith::SubIOp>(op1))
      if (subi.getLhs().getDefiningOp() == op2 &&
          isPositiveConstant(subi.getRhs()))
        return true;
    return false;
  };

  for (std::remove_const_t<decltype(size)> i = 0; i < size; i += 3) {
    // If both are loop invariant, skip to the next triple.
    if (mlir::isa_and_nonnull<fir::UndefOp>(ldTriples[i + 1].getDefiningOp()) &&
        mlir::isa_and_nonnull<fir::UndefOp>(stTriples[i + 1].getDefiningOp())) {
      // Unless either is a vector index, then be conservative.
      if (mlir::isa_and_nonnull<fir::UndefOp>(ldTriples[i].getDefiningOp()) ||
          mlir::isa_and_nonnull<fir::UndefOp>(stTriples[i].getDefiningOp()))
        return false;
      continue;
    }
    // If identical, skip to the next triple.
    if (ldTriples[i] == stTriples[i] && ldTriples[i + 1] == stTriples[i + 1] &&
        ldTriples[i + 2] == stTriples[i + 2])
      continue;
    // If ubound and lbound are the same with a constant offset, skip to the
    // next triple.
    if (displacedByConstant(ldTriples[i + 1], stTriples[i]) ||
        displacedByConstant(stTriples[i + 1], ldTriples[i]))
      continue;
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "detected non-overlapping slice ranges on " << ld
                          << " and " << st << ", which is not a conflict\n");
  return true;
}

/// Is there a conflict between the array value that was updated and to be
/// stored to `st` and the set of arrays loaded (`reach`) and used to compute
/// the updated value?
static bool conflictOnLoad(llvm::ArrayRef<mlir::Operation *> reach,
                           ArrayMergeStoreOp st) {
  mlir::Value load;
  mlir::Value addr = st.getMemref();
  const bool storeHasPointerType = hasPointerType(addr.getType());
  for (auto *op : reach)
    if (auto ld = mlir::dyn_cast<ArrayLoadOp>(op)) {
      mlir::Type ldTy = ld.getMemref().getType();
      if (ld.getMemref() == addr) {
        if (mutuallyExclusiveSliceRange(ld, st))
          continue;
        if (ld.getResult() != st.getOriginal())
          return true;
        if (load) {
          // TODO: extend this to allow checking if the first `load` and this
          // `ld` are mutually exclusive accesses but not identical.
          return true;
        }
        load = ld;
      } else if ((hasPointerType(ldTy) || storeHasPointerType)) {
        // TODO: Use target attribute to restrict this case further.
        // TODO: Check if types can also allow ruling out some cases. For now,
        // the fact that equivalences is using pointer attribute to enforce
        // aliasing is preventing any attempt to do so, and in general, it may
        // be wrong to use this if any of the types is a complex or a derived
        // for which it is possible to create a pointer to a part with a
        // different type than the whole, although this deserve some more
        // investigation because existing compiler behavior seem to diverge
        // here.
        return true;
      }
    }
  return false;
}

/// Is there an access vector conflict on the array being merged into? If the
/// access vectors diverge, then assume that there are potentially overlapping
/// loop-carried references.
static bool conflictOnMerge(llvm::ArrayRef<mlir::Operation *> mentions) {
  if (mentions.size() < 2)
    return false;
  llvm::SmallVector<mlir::Value> indices;
  LLVM_DEBUG(llvm::dbgs() << "check merge conflict on with " << mentions.size()
                          << " mentions on the list\n");
  bool valSeen = false;
  bool refSeen = false;
  for (auto *op : mentions) {
    llvm::SmallVector<mlir::Value> compareVector;
    if (auto u = mlir::dyn_cast<ArrayUpdateOp>(op)) {
      valSeen = true;
      if (indices.empty()) {
        indices = u.getIndices();
        continue;
      }
      compareVector = u.getIndices();
    } else if (auto f = mlir::dyn_cast<ArrayModifyOp>(op)) {
      valSeen = true;
      if (indices.empty()) {
        indices = f.getIndices();
        continue;
      }
      compareVector = f.getIndices();
    } else if (auto f = mlir::dyn_cast<ArrayFetchOp>(op)) {
      valSeen = true;
      if (indices.empty()) {
        indices = f.getIndices();
        continue;
      }
      compareVector = f.getIndices();
    } else if (auto f = mlir::dyn_cast<ArrayAccessOp>(op)) {
      refSeen = true;
      if (indices.empty()) {
        indices = f.getIndices();
        continue;
      }
      compareVector = f.getIndices();
    } else if (mlir::isa<ArrayAmendOp>(op)) {
      refSeen = true;
      continue;
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
  return valSeen && refSeen;
}

/// With element-by-reference semantics, an amended array with more than once
/// access to the same loaded array are conservatively considered a conflict.
/// Note: the array copy can still be eliminated in subsequent optimizations.
static bool conflictOnReference(llvm::ArrayRef<mlir::Operation *> mentions) {
  LLVM_DEBUG(llvm::dbgs() << "checking reference semantics " << mentions.size()
                          << '\n');
  if (mentions.size() < 3)
    return false;
  unsigned amendCount = 0;
  unsigned accessCount = 0;
  for (auto *op : mentions) {
    if (mlir::isa<ArrayAmendOp>(op) && ++amendCount > 1) {
      LLVM_DEBUG(llvm::dbgs() << "conflict: multiple amends of array value\n");
      return true;
    }
    if (mlir::isa<ArrayAccessOp>(op) && ++accessCount > 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "conflict: multiple accesses of array value\n");
      return true;
    }
    if (mlir::isa<ArrayFetchOp, ArrayUpdateOp, ArrayModifyOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "conflict: array value has both uses by-value and uses "
                    "by-reference. conservative assumption.\n");
      return true;
    }
  }
  return false;
}

static mlir::Operation *
amendingAccess(llvm::ArrayRef<mlir::Operation *> mentions) {
  for (auto *op : mentions)
    if (auto amend = mlir::dyn_cast<ArrayAmendOp>(op))
      return amend.getMemref().getDefiningOp();
  return {};
}

// Are either of types of conflicts present?
inline bool conflictDetected(llvm::ArrayRef<mlir::Operation *> reach,
                             llvm::ArrayRef<mlir::Operation *> accesses,
                             ArrayMergeStoreOp st) {
  return conflictOnLoad(reach, st) || conflictOnMerge(accesses);
}

// Assume that any call to a function that uses host-associations will be
// modifying the output array.
static bool
conservativeCallConflict(llvm::ArrayRef<mlir::Operation *> reaches) {
  return llvm::any_of(reaches, [](mlir::Operation *op) {
    if (auto call = mlir::dyn_cast<fir::CallOp>(op))
      if (auto callee =
              call.getCallableForCallee().dyn_cast<mlir::SymbolRefAttr>()) {
        auto module = op->getParentOfType<mlir::ModuleOp>();
        return hasHostAssociationArgument(
            module.lookupSymbol<mlir::func::FuncOp>(callee));
      }
    return false;
  });
}

/// Constructor of the array copy analysis.
/// This performs the analysis and saves the intermediate results.
void ArrayCopyAnalysis::construct(mlir::Operation *topLevelOp) {
  topLevelOp->walk([&](Operation *op) {
    if (auto st = mlir::dyn_cast<fir::ArrayMergeStoreOp>(op)) {
      llvm::SmallVector<mlir::Operation *> values;
      ReachCollector::reachingValues(values, st.getSequence());
      bool callConflict = conservativeCallConflict(values);
      llvm::SmallVector<mlir::Operation *> mentions;
      arrayMentions(mentions,
                    mlir::cast<ArrayLoadOp>(st.getOriginal().getDefiningOp()));
      bool conflict = conflictDetected(values, mentions, st);
      bool refConflict = conflictOnReference(mentions);
      if (callConflict || conflict || refConflict) {
        LLVM_DEBUG(llvm::dbgs()
                   << "CONFLICT: copies required for " << st << '\n'
                   << "   adding conflicts on: " << op << " and "
                   << st.getOriginal() << '\n');
        conflicts.insert(op);
        conflicts.insert(st.getOriginal().getDefiningOp());
        if (auto *access = amendingAccess(mentions))
          amendAccesses.insert(access);
      }
      auto *ld = st.getOriginal().getDefiningOp();
      LLVM_DEBUG(llvm::dbgs()
                 << "map: adding {" << *ld << " -> " << st << "}\n");
      useMap.insert({ld, op});
    } else if (auto load = mlir::dyn_cast<ArrayLoadOp>(op)) {
      llvm::SmallVector<mlir::Operation *> mentions;
      arrayMentions(mentions, load);
      LLVM_DEBUG(llvm::dbgs() << "process load: " << load
                              << ", mentions: " << mentions.size() << '\n');
      for (auto *acc : mentions) {
        LLVM_DEBUG(llvm::dbgs() << " mention: " << *acc << '\n');
        if (mlir::isa<ArrayAccessOp, ArrayAmendOp, ArrayFetchOp, ArrayUpdateOp,
                      ArrayModifyOp>(acc)) {
          if (useMap.count(acc)) {
            mlir::emitError(
                load.getLoc(),
                "The parallel semantics of multiple array_merge_stores per "
                "array_load are not supported.");
            continue;
          }
          LLVM_DEBUG(llvm::dbgs()
                     << "map: adding {" << *acc << "} -> {" << load << "}\n");
          useMap.insert({acc, op});
        }
      }
    }
  });
}

//===----------------------------------------------------------------------===//
// Conversions for converting out of array value form.
//===----------------------------------------------------------------------===//

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
  auto eleTy = unwrapSequenceType(unwrapPassByRefType(ty));
  // FIXME: keep ptr/heap/ref information.
  return ReferenceType::get(eleTy);
}

// Extract extents from the ShapeOp/ShapeShiftOp into the result vector.
static bool getAdjustedExtents(mlir::Location loc,
                               mlir::PatternRewriter &rewriter,
                               ArrayLoadOp arrLoad,
                               llvm::SmallVectorImpl<mlir::Value> &result,
                               mlir::Value shape) {
  bool copyUsingSlice = false;
  auto *shapeOp = shape.getDefiningOp();
  if (auto s = mlir::dyn_cast_or_null<ShapeOp>(shapeOp)) {
    auto e = s.getExtents();
    result.insert(result.end(), e.begin(), e.end());
  } else if (auto s = mlir::dyn_cast_or_null<ShapeShiftOp>(shapeOp)) {
    auto e = s.getExtents();
    result.insert(result.end(), e.begin(), e.end());
  } else {
    emitFatalError(loc, "not a fir.shape/fir.shape_shift op");
  }
  auto idxTy = rewriter.getIndexType();
  if (factory::isAssumedSize(result)) {
    // Use slice information to compute the extent of the column.
    auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value size = one;
    if (mlir::Value sliceArg = arrLoad.getSlice()) {
      if (auto sliceOp =
              mlir::dyn_cast_or_null<SliceOp>(sliceArg.getDefiningOp())) {
        auto triples = sliceOp.getTriples();
        const std::size_t tripleSize = triples.size();
        auto module = arrLoad->getParentOfType<mlir::ModuleOp>();
        FirOpBuilder builder(rewriter, getKindMapping(module));
        size = builder.genExtentFromTriplet(loc, triples[tripleSize - 3],
                                            triples[tripleSize - 2],
                                            triples[tripleSize - 1], idxTy);
        copyUsingSlice = true;
      }
    }
    result[result.size() - 1] = size;
  }
  return copyUsingSlice;
}

/// Place the extents of the array load, \p arrLoad, into \p result and
/// return a ShapeOp or ShapeShiftOp with the same extents. If \p arrLoad is
/// loading a `!fir.box`, code will be generated to read the extents from the
/// boxed value, and the retunred shape Op will be built with the extents read
/// from the box. Otherwise, the extents will be extracted from the ShapeOp (or
/// ShapeShiftOp) argument of \p arrLoad. \p copyUsingSlice will be set to true
/// if slicing of the output array is to be done in the copy-in/copy-out rather
/// than in the elemental computation step.
static mlir::Value getOrReadExtentsAndShapeOp(
    mlir::Location loc, mlir::PatternRewriter &rewriter, ArrayLoadOp arrLoad,
    llvm::SmallVectorImpl<mlir::Value> &result, bool &copyUsingSlice) {
  assert(result.empty());
  if (arrLoad->hasAttr(fir::getOptionalAttrName()))
    fir::emitFatalError(
        loc, "shapes from array load of OPTIONAL arrays must not be used");
  if (auto boxTy = arrLoad.getMemref().getType().dyn_cast<BoxType>()) {
    auto rank =
        dyn_cast_ptrOrBoxEleTy(boxTy).cast<SequenceType>().getDimension();
    auto idxTy = rewriter.getIndexType();
    for (decltype(rank) dim = 0; dim < rank; ++dim) {
      auto dimVal = rewriter.create<mlir::arith::ConstantIndexOp>(loc, dim);
      auto dimInfo = rewriter.create<BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                                arrLoad.getMemref(), dimVal);
      result.emplace_back(dimInfo.getResult(1));
    }
    if (!arrLoad.getShape()) {
      auto shapeType = ShapeType::get(rewriter.getContext(), rank);
      return rewriter.create<ShapeOp>(loc, shapeType, result);
    }
    auto shiftOp = arrLoad.getShape().getDefiningOp<ShiftOp>();
    auto shapeShiftType = ShapeShiftType::get(rewriter.getContext(), rank);
    llvm::SmallVector<mlir::Value> shapeShiftOperands;
    for (auto [lb, extent] : llvm::zip(shiftOp.getOrigins(), result)) {
      shapeShiftOperands.push_back(lb);
      shapeShiftOperands.push_back(extent);
    }
    return rewriter.create<ShapeShiftOp>(loc, shapeShiftType,
                                         shapeShiftOperands);
  }
  copyUsingSlice =
      getAdjustedExtents(loc, rewriter, arrLoad, result, arrLoad.getShape());
  return arrLoad.getShape();
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

static mlir::Value getCharacterLen(mlir::Location loc, FirOpBuilder &builder,
                                   ArrayLoadOp load, CharacterType charTy) {
  auto charLenTy = builder.getCharacterLengthType();
  if (charTy.hasDynamicLen()) {
    if (load.getMemref().getType().isa<BoxType>()) {
      // The loaded array is an emboxed value. Get the CHARACTER length from
      // the box value.
      auto eleSzInBytes =
          builder.create<BoxEleSizeOp>(loc, charLenTy, load.getMemref());
      auto kindSize =
          builder.getKindMap().getCharacterBitsize(charTy.getFKind());
      auto kindByteSize =
          builder.createIntegerConstant(loc, charLenTy, kindSize / 8);
      return builder.create<mlir::arith::DivSIOp>(loc, eleSzInBytes,
                                                  kindByteSize);
    }
    // The loaded array is a (set of) unboxed values. If the CHARACTER's
    // length is not a constant, it must be provided as a type parameter to
    // the array_load.
    auto typeparams = load.getTypeparams();
    assert(typeparams.size() > 0 && "expected type parameters on array_load");
    return typeparams.back();
  }
  // The typical case: the length of the CHARACTER is a compile-time
  // constant that is encoded in the type information.
  return builder.createIntegerConstant(loc, charLenTy, charTy.getLen());
}
/// Generate a shallow array copy. This is used for both copy-in and copy-out.
template <bool CopyIn>
void genArrayCopy(mlir::Location loc, mlir::PatternRewriter &rewriter,
                  mlir::Value dst, mlir::Value src, mlir::Value shapeOp,
                  mlir::Value sliceOp, ArrayLoadOp arrLoad) {
  auto insPt = rewriter.saveInsertionPoint();
  llvm::SmallVector<mlir::Value> indices;
  llvm::SmallVector<mlir::Value> extents;
  bool copyUsingSlice =
      getAdjustedExtents(loc, rewriter, arrLoad, extents, shapeOp);
  auto idxTy = rewriter.getIndexType();
  // Build loop nest from column to row.
  for (auto sh : llvm::reverse(extents)) {
    auto ubi = rewriter.create<ConvertOp>(loc, idxTy, sh);
    auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto ub = rewriter.create<mlir::arith::SubIOp>(loc, idxTy, ubi, one);
    auto loop = rewriter.create<DoLoopOp>(loc, zero, ub, one);
    rewriter.setInsertionPointToStart(loop.getBody());
    indices.push_back(loop.getInductionVar());
  }
  // Reverse the indices so they are in column-major order.
  std::reverse(indices.begin(), indices.end());
  auto typeparams = arrLoad.getTypeparams();
  auto fromAddr = rewriter.create<ArrayCoorOp>(
      loc, getEleTy(src.getType()), src, shapeOp,
      CopyIn && copyUsingSlice ? sliceOp : mlir::Value{},
      factory::originateIndices(loc, rewriter, src.getType(), shapeOp, indices),
      typeparams);
  auto toAddr = rewriter.create<ArrayCoorOp>(
      loc, getEleTy(dst.getType()), dst, shapeOp,
      !CopyIn && copyUsingSlice ? sliceOp : mlir::Value{},
      factory::originateIndices(loc, rewriter, dst.getType(), shapeOp, indices),
      typeparams);
  auto eleTy = unwrapSequenceType(unwrapPassByRefType(dst.getType()));
  auto module = toAddr->getParentOfType<mlir::ModuleOp>();
  FirOpBuilder builder(rewriter, getKindMapping(module));
  // Copy from (to) object to (from) temp copy of same object.
  if (auto charTy = eleTy.dyn_cast<CharacterType>()) {
    auto len = getCharacterLen(loc, builder, arrLoad, charTy);
    CharBoxValue toChar(toAddr, len);
    CharBoxValue fromChar(fromAddr, len);
    factory::genScalarAssignment(builder, loc, toChar, fromChar);
  } else {
    if (hasDynamicSize(eleTy))
      TODO(loc, "copy element of dynamic size");
    factory::genScalarAssignment(builder, loc, toAddr, fromAddr);
  }
  rewriter.restoreInsertionPoint(insPt);
}

/// The array load may be either a boxed or unboxed value. If the value is
/// boxed, we read the type parameters from the boxed value.
static llvm::SmallVector<mlir::Value>
genArrayLoadTypeParameters(mlir::Location loc, mlir::PatternRewriter &rewriter,
                           ArrayLoadOp load) {
  if (load.getTypeparams().empty()) {
    auto eleTy =
        unwrapSequenceType(unwrapPassByRefType(load.getMemref().getType()));
    if (hasDynamicSize(eleTy)) {
      if (auto charTy = eleTy.dyn_cast<CharacterType>()) {
        assert(load.getMemref().getType().isa<BoxType>());
        auto module = load->getParentOfType<mlir::ModuleOp>();
        FirOpBuilder builder(rewriter, getKindMapping(module));
        return {getCharacterLen(loc, builder, load, charTy)};
      }
      TODO(loc, "unhandled dynamic type parameters");
    }
    return {};
  }
  return load.getTypeparams();
}

static llvm::SmallVector<mlir::Value>
findNonconstantExtents(mlir::Type memrefTy,
                       llvm::ArrayRef<mlir::Value> extents) {
  llvm::SmallVector<mlir::Value> nce;
  auto arrTy = unwrapPassByRefType(memrefTy);
  auto seqTy = arrTy.cast<SequenceType>();
  for (auto [s, x] : llvm::zip(seqTy.getShape(), extents))
    if (s == SequenceType::getUnknownExtent())
      nce.emplace_back(x);
  if (extents.size() > seqTy.getShape().size())
    for (auto x : extents.drop_front(seqTy.getShape().size()))
      nce.emplace_back(x);
  return nce;
}

/// Allocate temporary storage for an ArrayLoadOp \load and initialize any
/// allocatable direct components of the array elements with an unallocated
/// status. Returns the temporary address as well as a callback to generate the
/// temporary clean-up once it has been used. The clean-up will take care of
/// deallocating all the element allocatable components that may have been
/// allocated while using the temporary.
static std::pair<mlir::Value,
                 std::function<void(mlir::PatternRewriter &rewriter)>>
allocateArrayTemp(mlir::Location loc, mlir::PatternRewriter &rewriter,
                  ArrayLoadOp load, llvm::ArrayRef<mlir::Value> extents,
                  mlir::Value shape) {
  mlir::Type baseType = load.getMemref().getType();
  llvm::SmallVector<mlir::Value> nonconstantExtents =
      findNonconstantExtents(baseType, extents);
  llvm::SmallVector<mlir::Value> typeParams =
      genArrayLoadTypeParameters(loc, rewriter, load);
  mlir::Value allocmem = rewriter.create<AllocMemOp>(
      loc, dyn_cast_ptrOrBoxEleTy(baseType), typeParams, nonconstantExtents);
  mlir::Type eleType =
      fir::unwrapSequenceType(fir::unwrapPassByRefType(baseType));
  if (fir::isRecordWithAllocatableMember(eleType)) {
    // The allocatable component descriptors need to be set to a clean
    // deallocated status before anything is done with them.
    mlir::Value box = rewriter.create<fir::EmboxOp>(
        loc, fir::BoxType::get(baseType), allocmem, shape,
        /*slice=*/mlir::Value{}, typeParams);
    auto module = load->getParentOfType<mlir::ModuleOp>();
    FirOpBuilder builder(rewriter, getKindMapping(module));
    runtime::genDerivedTypeInitialize(builder, loc, box);
    // Any allocatable component that may have been allocated must be
    // deallocated during the clean-up.
    auto cleanup = [=](mlir::PatternRewriter &r) {
      FirOpBuilder builder(r, getKindMapping(module));
      runtime::genDerivedTypeDestroy(builder, loc, box);
      r.create<FreeMemOp>(loc, allocmem);
    };
    return {allocmem, cleanup};
  }
  auto cleanup = [=](mlir::PatternRewriter &r) {
    r.create<FreeMemOp>(loc, allocmem);
  };
  return {allocmem, cleanup};
}

namespace {
/// Conversion of fir.array_update and fir.array_modify Ops.
/// If there is a conflict for the update, then we need to perform a
/// copy-in/copy-out to preserve the original values of the array. If there is
/// no conflict, then it is save to eschew making any copies.
template <typename ArrayOp>
class ArrayUpdateConversionBase : public mlir::OpRewritePattern<ArrayOp> {
public:
  // TODO: Implement copy/swap semantics?
  explicit ArrayUpdateConversionBase(mlir::MLIRContext *ctx,
                                     const ArrayCopyAnalysis &a,
                                     const OperationUseMapT &m)
      : mlir::OpRewritePattern<ArrayOp>{ctx}, analysis{a}, useMap{m} {}

  /// The array_access, \p access, is to be to a cloned copy due to a potential
  /// conflict. Uses copy-in/copy-out semantics and not copy/swap.
  mlir::Value referenceToClone(mlir::Location loc,
                               mlir::PatternRewriter &rewriter,
                               ArrayOp access) const {
    LLVM_DEBUG(llvm::dbgs()
               << "generating copy-in/copy-out loops for " << access << '\n');
    auto *op = access.getOperation();
    auto *loadOp = useMap.lookup(op);
    auto load = mlir::cast<ArrayLoadOp>(loadOp);
    auto eleTy = access.getType();
    rewriter.setInsertionPoint(loadOp);
    // Copy in.
    llvm::SmallVector<mlir::Value> extents;
    bool copyUsingSlice = false;
    auto shapeOp = getOrReadExtentsAndShapeOp(loc, rewriter, load, extents,
                                              copyUsingSlice);
    auto [allocmem, genTempCleanUp] =
        allocateArrayTemp(loc, rewriter, load, extents, shapeOp);
    genArrayCopy</*copyIn=*/true>(load.getLoc(), rewriter, allocmem,
                                  load.getMemref(), shapeOp, load.getSlice(),
                                  load);
    // Generate the reference for the access.
    rewriter.setInsertionPoint(op);
    auto coor =
        genCoorOp(rewriter, loc, getEleTy(load.getType()), eleTy, allocmem,
                  shapeOp, copyUsingSlice ? mlir::Value{} : load.getSlice(),
                  access.getIndices(), load.getTypeparams(),
                  access->hasAttr(factory::attrFortranArrayOffsets()));
    // Copy out.
    auto *storeOp = useMap.lookup(loadOp);
    auto store = mlir::cast<ArrayMergeStoreOp>(storeOp);
    rewriter.setInsertionPoint(storeOp);
    // Copy out.
    genArrayCopy</*copyIn=*/false>(store.getLoc(), rewriter, store.getMemref(),
                                   allocmem, shapeOp, store.getSlice(), load);
    genTempCleanUp(rewriter);
    return coor;
  }

  /// Copy the RHS element into the LHS and insert copy-in/copy-out between a
  /// temp and the LHS if the analysis found potential overlaps between the RHS
  /// and LHS arrays. The element copy generator must be provided in \p
  /// assignElement. \p update must be the ArrayUpdateOp or the ArrayModifyOp.
  /// Returns the address of the LHS element inside the loop and the LHS
  /// ArrayLoad result.
  std::pair<mlir::Value, mlir::Value>
  materializeAssignment(mlir::Location loc, mlir::PatternRewriter &rewriter,
                        ArrayOp update,
                        const std::function<void(mlir::Value)> &assignElement,
                        mlir::Type lhsEltRefType) const {
    auto *op = update.getOperation();
    auto *loadOp = useMap.lookup(op);
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
      bool copyUsingSlice = false;
      auto shapeOp = getOrReadExtentsAndShapeOp(loc, rewriter, load, extents,
                                                copyUsingSlice);
      auto [allocmem, genTempCleanUp] =
          allocateArrayTemp(loc, rewriter, load, extents, shapeOp);

      genArrayCopy</*copyIn=*/true>(load.getLoc(), rewriter, allocmem,
                                    load.getMemref(), shapeOp, load.getSlice(),
                                    load);
      rewriter.setInsertionPoint(op);
      auto coor = genCoorOp(
          rewriter, loc, getEleTy(load.getType()), lhsEltRefType, allocmem,
          shapeOp, copyUsingSlice ? mlir::Value{} : load.getSlice(),
          update.getIndices(), load.getTypeparams(),
          update->hasAttr(factory::attrFortranArrayOffsets()));
      assignElement(coor);
      auto *storeOp = useMap.lookup(loadOp);
      auto store = mlir::cast<ArrayMergeStoreOp>(storeOp);
      rewriter.setInsertionPoint(storeOp);
      // Copy out.
      genArrayCopy</*copyIn=*/false>(store.getLoc(), rewriter,
                                     store.getMemref(), allocmem, shapeOp,
                                     store.getSlice(), load);
      genTempCleanUp(rewriter);
      return {coor, load.getResult()};
    }
    // Otherwise, when there is no conflict (a possible loop-carried
    // dependence), the lhs array can be updated in place.
    LLVM_DEBUG(llvm::outs() << "No, conflict wasn't found\n");
    rewriter.setInsertionPoint(op);
    auto coorTy = getEleTy(load.getType());
    auto coor = genCoorOp(rewriter, loc, coorTy, lhsEltRefType,
                          load.getMemref(), load.getShape(), load.getSlice(),
                          update.getIndices(), load.getTypeparams(),
                          update->hasAttr(factory::attrFortranArrayOffsets()));
    assignElement(coor);
    return {coor, load.getResult()};
  }

protected:
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
      auto input = update.getMerge();
      if (auto inEleTy = dyn_cast_ptrEleTy(input.getType())) {
        emitFatalError(loc, "array_update on references not supported");
      } else {
        rewriter.create<fir::StoreOp>(loc, input, coor);
      }
    };
    auto lhsEltRefType = toRefType(update.getMerge().getType());
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
    auto coor =
        genCoorOp(rewriter, loc, getEleTy(load.getType()),
                  toRefType(fetch.getType()), load.getMemref(), load.getShape(),
                  load.getSlice(), fetch.getIndices(), load.getTypeparams(),
                  fetch->hasAttr(factory::attrFortranArrayOffsets()));
    if (isa_ref_type(fetch.getType()))
      rewriter.replaceOp(fetch, coor);
    else
      rewriter.replaceOpWithNewOp<fir::LoadOp>(fetch, coor);
    return mlir::success();
  }

private:
  const OperationUseMapT &useMap;
};

/// As array_access op is like an array_fetch op, except that it does not imply
/// a load op. (It operates in the reference domain.)
class ArrayAccessConversion : public ArrayUpdateConversionBase<ArrayAccessOp> {
public:
  explicit ArrayAccessConversion(mlir::MLIRContext *ctx,
                                 const ArrayCopyAnalysis &a,
                                 const OperationUseMapT &m)
      : ArrayUpdateConversionBase{ctx, a, m} {}

  mlir::LogicalResult
  matchAndRewrite(ArrayAccessOp access,
                  mlir::PatternRewriter &rewriter) const override {
    auto *op = access.getOperation();
    auto loc = access.getLoc();
    if (analysis.inAmendAccessSet(op)) {
      // This array_access is associated with an array_amend and there is a
      // conflict. Make a copy to store into.
      auto result = referenceToClone(loc, rewriter, access);
      access.replaceAllUsesWith(result);
      rewriter.replaceOp(access, result);
      return mlir::success();
    }
    rewriter.setInsertionPoint(op);
    auto load = mlir::cast<ArrayLoadOp>(useMap.lookup(op));
    auto coor = genCoorOp(rewriter, loc, getEleTy(load.getType()),
                          toRefType(access.getType()), load.getMemref(),
                          load.getShape(), load.getSlice(), access.getIndices(),
                          load.getTypeparams(),
                          access->hasAttr(factory::attrFortranArrayOffsets()));
    rewriter.replaceOp(access, coor);
    return mlir::success();
  }
};

/// An array_amend op is a marker to record which array access is being used to
/// update an array value. After this pass runs, an array_amend has no
/// semantics. We rewrite these to undefined values here to remove them while
/// preserving SSA form.
class ArrayAmendConversion : public mlir::OpRewritePattern<ArrayAmendOp> {
public:
  explicit ArrayAmendConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(ArrayAmendOp amend,
                  mlir::PatternRewriter &rewriter) const override {
    auto *op = amend.getOperation();
    rewriter.setInsertionPoint(op);
    auto loc = amend.getLoc();
    auto undef = rewriter.create<UndefOp>(loc, amend.getType());
    rewriter.replaceOp(amend, undef.getResult());
    return mlir::success();
  }
};

class ArrayValueCopyConverter
    : public ArrayValueCopyBase<ArrayValueCopyConverter> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "\n\narray-value-copy pass on function '"
                            << func.getName() << "'\n");
    auto *context = &getContext();

    // Perform the conflict analysis.
    const auto &analysis = getAnalysis<ArrayCopyAnalysis>();
    const auto &useMap = analysis.getUseMap();

    mlir::RewritePatternSet patterns1(context);
    patterns1.insert<ArrayFetchConversion>(context, useMap);
    patterns1.insert<ArrayUpdateConversion>(context, analysis, useMap);
    patterns1.insert<ArrayModifyConversion>(context, analysis, useMap);
    patterns1.insert<ArrayAccessConversion>(context, analysis, useMap);
    patterns1.insert<ArrayAmendConversion>(context);
    mlir::ConversionTarget target(*context);
    target.addLegalDialect<FIROpsDialect, mlir::scf::SCFDialect,
                           mlir::arith::ArithmeticDialect,
                           mlir::func::FuncDialect>();
    target.addIllegalOp<ArrayAccessOp, ArrayAmendOp, ArrayFetchOp,
                        ArrayUpdateOp, ArrayModifyOp>();
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
