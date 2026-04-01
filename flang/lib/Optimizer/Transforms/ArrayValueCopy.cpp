//===-- ArrayValueCopy.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Factory.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "aiir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

namespace fir {
#define GEN_PASS_DEF_ARRAYVALUECOPY
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-array-value-copy"

using namespace fir;
using namespace aiir;

using OperationUseMapT = llvm::DenseMap<aiir::Operation *, aiir::Operation *>;

namespace {

/// Array copy analysis.
/// Perform an interference analysis between array values.
///
/// Lowering will generate a sequence of the following form.
/// ```aiir
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
class ArrayCopyAnalysisBase {
public:
  using ConflictSetT = llvm::SmallPtrSet<aiir::Operation *, 16>;
  using UseSetT = llvm::SmallPtrSet<aiir::OpOperand *, 8>;
  using LoadMapSetsT = llvm::DenseMap<aiir::Operation *, UseSetT>;
  using AmendAccessSetT = llvm::SmallPtrSet<aiir::Operation *, 4>;

  ArrayCopyAnalysisBase(aiir::Operation *op, bool optimized)
      : operation{op}, optimizeConflicts(optimized) {
    construct(op);
  }
  virtual ~ArrayCopyAnalysisBase() = default;

  aiir::Operation *getOperation() const { return operation; }

  /// Return true iff the `array_merge_store` has potential conflicts.
  bool hasPotentialConflict(aiir::Operation *op) const {
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
  bool inAmendAccessSet(aiir::Operation *op) const {
    return amendAccesses.count(op);
  }

  /// For ArrayLoad `load`, return the transitive set of all OpOperands.
  UseSetT getLoadUseSet(aiir::Operation *load) const {
    assert(loadMapSets.count(load) && "analysis missed an array load?");
    return loadMapSets.lookup(load);
  }

  void arrayMentions(llvm::SmallVectorImpl<aiir::Operation *> &mentions,
                     ArrayLoadOp load);

private:
  void construct(aiir::Operation *topLevelOp);

  aiir::Operation *operation; // operation that analysis ran upon
  ConflictSetT conflicts;     // set of conflicts (loads and merge stores)
  OperationUseMapT useMap;
  LoadMapSetsT loadMapSets;
  // Set of array_access ops associated with array_amend ops.
  AmendAccessSetT amendAccesses;
  bool optimizeConflicts;
};

// Optimized array copy analysis that takes into account Fortran
// variable attributes to prove that no conflict is possible
// and reduce the number of temporary arrays.
class ArrayCopyAnalysisOptimized : public ArrayCopyAnalysisBase {
public:
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ArrayCopyAnalysisOptimized)

  ArrayCopyAnalysisOptimized(aiir::Operation *op)
      : ArrayCopyAnalysisBase(op, /*optimized=*/true) {}
};

// Unoptimized array copy analysis used at O0.
class ArrayCopyAnalysis : public ArrayCopyAnalysisBase {
public:
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ArrayCopyAnalysis)

  ArrayCopyAnalysis(aiir::Operation *op)
      : ArrayCopyAnalysisBase(op, /*optimized=*/false) {}
};
} // namespace

namespace {
/// Helper class to collect all array operations that produced an array value.
class ReachCollector {
public:
  ReachCollector(llvm::SmallVectorImpl<aiir::Operation *> &reach,
                 aiir::Region *loopRegion)
      : reach{reach}, loopRegion{loopRegion} {}

  void collectArrayMentionFrom(aiir::Operation *op, aiir::ValueRange range) {
    if (range.empty()) {
      collectArrayMentionFrom(op, aiir::Value{});
      return;
    }
    for (aiir::Value v : range)
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
                       aiir::Block *block) {
    for (auto &op : *block) {
      if (auto access = aiir::dyn_cast<ArrayAccessOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "adding access: " << access << '\n');
        result.push_back(access);
        continue;
      }
      for (auto &region : op.getRegions())
        for (auto &bb : region.getBlocks())
          collectAccesses(result, &bb);
    }
  }

  void collectArrayMentionFrom(aiir::Operation *op, aiir::Value val) {
    // `val` is defined by an Op, process the defining Op.
    // If `val` is defined by a region containing Op, we want to drill down
    // and through that Op's region(s).
    LLVM_DEBUG(llvm::dbgs() << "popset: " << *op << '\n');
    auto popFn = [&](auto rop) {
      assert(val && "op must have a result value");
      auto resNum = aiir::cast<aiir::OpResult>(val).getResultNumber();
      llvm::SmallVector<aiir::Value> results;
      rop.resultToSourceOps(results, resNum);
      for (auto u : results)
        collectArrayMentionFrom(u);
    };
    if (auto rop = aiir::dyn_cast<DoLoopOp>(op)) {
      popFn(rop);
      return;
    }
    if (auto rop = aiir::dyn_cast<IterWhileOp>(op)) {
      popFn(rop);
      return;
    }
    if (auto rop = aiir::dyn_cast<fir::IfOp>(op)) {
      popFn(rop);
      return;
    }
    if (auto box = aiir::dyn_cast<EmboxOp>(op)) {
      for (auto *user : box.getMemref().getUsers())
        if (user != op)
          collectArrayMentionFrom(user, user->getResults());
      return;
    }
    if (auto mergeStore = aiir::dyn_cast<ArrayMergeStoreOp>(op)) {
      if (opIsInsideLoops(mergeStore))
        collectArrayMentionFrom(mergeStore.getSequence());
      return;
    }

    if (aiir::isa<AllocaOp, AllocMemOp>(op)) {
      // Look for any stores inside the loops, and collect an array operation
      // that produced the value being stored to it.
      for (auto *user : op->getUsers())
        if (auto store = aiir::dyn_cast<fir::StoreOp>(user))
          if (opIsInsideLoops(store))
            collectArrayMentionFrom(store.getValue());
      return;
    }

    // Scan the uses of amend's memref
    if (auto amend = aiir::dyn_cast<ArrayAmendOp>(op)) {
      reach.push_back(op);
      llvm::SmallVector<ArrayAccessOp> accesses;
      collectAccesses(accesses, op->getBlock());
      for (auto access : accesses)
        collectArrayMentionFrom(access.getResult());
    }

    // Otherwise, Op does not contain a region so just chase its operands.
    if (aiir::isa<ArrayAccessOp, ArrayLoadOp, ArrayUpdateOp, ArrayModifyOp,
                  ArrayFetchOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "add " << *op << " to reachable set\n");
      reach.push_back(op);
    }

    // Include all array_access ops using an array_load.
    if (auto arrLd = aiir::dyn_cast<ArrayLoadOp>(op))
      for (auto *user : arrLd.getResult().getUsers())
        if (aiir::isa<ArrayAccessOp>(user)) {
          LLVM_DEBUG(llvm::dbgs() << "add " << *user << " to reachable set\n");
          reach.push_back(user);
        }

    // Array modify assignment is performed on the result. So the analysis must
    // look at the what is done with the result.
    if (aiir::isa<ArrayModifyOp>(op))
      for (auto *user : op->getResult(0).getUsers())
        followUsers(user);

    if (aiir::isa<fir::CallOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "add " << *op << " to reachable set\n");
      reach.push_back(op);
    }

    for (auto u : op->getOperands())
      collectArrayMentionFrom(u);
  }

  void collectArrayMentionFrom(aiir::BlockArgument ba) {
    auto *parent = ba.getOwner()->getParentOp();
    // If inside an Op holding a region, the block argument corresponds to an
    // argument passed to the containing Op.
    auto popFn = [&](auto rop) {
      collectArrayMentionFrom(rop.blockArgToSourceOp(ba.getArgNumber()));
    };
    if (auto rop = aiir::dyn_cast<DoLoopOp>(parent)) {
      popFn(rop);
      return;
    }
    if (auto rop = aiir::dyn_cast<IterWhileOp>(parent)) {
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
  void collectArrayMentionFrom(aiir::Value val) {
    if (!val || visited.contains(val))
      return;
    visited.insert(val);

    // Process a block argument.
    if (auto ba = aiir::dyn_cast<aiir::BlockArgument>(val)) {
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
  static void reachingValues(llvm::SmallVectorImpl<aiir::Operation *> &reach,
                             aiir::Value seq) {
    reach.clear();
    aiir::Region *loopRegion = nullptr;
    if (auto doLoop = aiir::dyn_cast_or_null<DoLoopOp>(seq.getDefiningOp()))
      loopRegion = &doLoop->getRegion(0);
    ReachCollector collector(reach, loopRegion);
    collector.collectArrayMentionFrom(seq);
  }

private:
  /// Is \op inside the loop nest region ?
  /// FIXME: replace this structural dependence with graph properties.
  bool opIsInsideLoops(aiir::Operation *op) const {
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
  void followUsers(aiir::Operation *op) {
    for (auto userOperand : op->getOperands())
      collectArrayMentionFrom(userOperand);
    // Go through potential converts/coordinate_op.
    for (auto indirectUser : op->getUsers())
      followUsers(indirectUser);
  }

  llvm::SmallVectorImpl<aiir::Operation *> &reach;
  llvm::SmallPtrSet<aiir::Value, 16> visited;
  /// Region of the loops nest that produced the array value.
  aiir::Region *loopRegion;
};
} // namespace

/// Find all the array operations that access the array value that is loaded by
/// the array load operation, `load`.
void ArrayCopyAnalysisBase::arrayMentions(
    llvm::SmallVectorImpl<aiir::Operation *> &mentions, ArrayLoadOp load) {
  mentions.clear();
  auto lmIter = loadMapSets.find(load);
  if (lmIter != loadMapSets.end()) {
    for (auto *opnd : lmIter->second) {
      auto *owner = opnd->getOwner();
      if (aiir::isa<ArrayAccessOp, ArrayAmendOp, ArrayFetchOp, ArrayUpdateOp,
                    ArrayModifyOp>(owner))
        mentions.push_back(owner);
    }
    return;
  }

  UseSetT visited;
  llvm::SmallVector<aiir::OpOperand *> queue; // uses of ArrayLoad[orig]

  auto appendToQueue = [&](aiir::Value val) {
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
    aiir::OpOperand *operand = queue.pop_back_val();
    aiir::Operation *owner = operand->getOwner();
    if (!owner)
      continue;
    auto structuredLoop = [&](auto ro) {
      if (auto blockArg = ro.iterArgToBlockArg(operand->get())) {
        int64_t arg = blockArg.getArgNumber();
        aiir::Value output = ro.getResult(ro.getFinalValue() ? arg : arg - 1);
        appendToQueue(output);
        appendToQueue(blockArg);
      }
    };
    // TODO: this need to be updated to use the control-flow interface.
    auto branchOp = [&](aiir::Block *dest, OperandRange operands) {
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
    if (auto ro = aiir::dyn_cast<DoLoopOp>(owner)) {
      structuredLoop(ro);
    } else if (auto ro = aiir::dyn_cast<IterWhileOp>(owner)) {
      structuredLoop(ro);
    } else if (auto rs = aiir::dyn_cast<ResultOp>(owner)) {
      // Thread any uses of fir.if that return the marked array value.
      aiir::Operation *parent = rs->getParentRegion()->getParentOp();
      if (auto ifOp = aiir::dyn_cast<fir::IfOp>(parent))
        appendToQueue(ifOp.getResult(operand->getOperandNumber()));
    } else if (aiir::isa<ArrayFetchOp>(owner)) {
      // Keep track of array value fetches.
      LLVM_DEBUG(llvm::dbgs()
                 << "add fetch {" << *owner << "} to array value set\n");
      mentions.push_back(owner);
    } else if (auto update = aiir::dyn_cast<ArrayUpdateOp>(owner)) {
      // Keep track of array value updates and thread the return value uses.
      LLVM_DEBUG(llvm::dbgs()
                 << "add update {" << *owner << "} to array value set\n");
      mentions.push_back(owner);
      appendToQueue(update.getResult());
    } else if (auto update = aiir::dyn_cast<ArrayModifyOp>(owner)) {
      // Keep track of array value modification and thread the return value
      // uses.
      LLVM_DEBUG(llvm::dbgs()
                 << "add modify {" << *owner << "} to array value set\n");
      mentions.push_back(owner);
      appendToQueue(update.getResult(1));
    } else if (auto mention = aiir::dyn_cast<ArrayAccessOp>(owner)) {
      mentions.push_back(owner);
    } else if (auto amend = aiir::dyn_cast<ArrayAmendOp>(owner)) {
      mentions.push_back(owner);
      appendToQueue(amend.getResult());
    } else if (auto br = aiir::dyn_cast<aiir::cf::BranchOp>(owner)) {
      branchOp(br.getDest(), br.getDestOperands());
    } else if (auto br = aiir::dyn_cast<aiir::cf::CondBranchOp>(owner)) {
      branchOp(br.getTrueDest(), br.getTrueOperands());
      branchOp(br.getFalseDest(), br.getFalseOperands());
    } else if (aiir::isa<ArrayMergeStoreOp>(owner)) {
      // do nothing
    } else {
      llvm::report_fatal_error("array value reached unexpected op");
    }
  }
  loadMapSets.insert({load, visited});
}

static bool hasPointerType(aiir::Type type) {
  if (auto boxTy = aiir::dyn_cast<BoxType>(type))
    type = boxTy.getEleTy();
  return aiir::isa<fir::PointerType>(type);
}

// This is a NF performance hack. It makes a simple test that the slices of the
// load, \p ld, and the merge store, \p st, are trivially mutually exclusive.
static bool mutuallyExclusiveSliceRange(ArrayLoadOp ld, ArrayMergeStoreOp st) {
  // If the same array_load, then no further testing is warranted.
  if (ld.getResult() == st.getOriginal())
    return false;

  auto getSliceOp = [](aiir::Value val) -> SliceOp {
    if (!val)
      return {};
    auto sliceOp = aiir::dyn_cast_or_null<SliceOp>(val.getDefiningOp());
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

  auto displacedByConstant = [](aiir::Value v1, aiir::Value v2) {
    auto removeConvert = [](aiir::Value v) -> aiir::Operation * {
      auto *op = v.getDefiningOp();
      while (auto conv = aiir::dyn_cast_or_null<ConvertOp>(op))
        op = conv.getValue().getDefiningOp();
      return op;
    };

    auto isPositiveConstant = [](aiir::Value v) -> bool {
      if (auto conOp =
              aiir::dyn_cast<aiir::arith::ConstantOp>(v.getDefiningOp()))
        if (auto iattr = aiir::dyn_cast<aiir::IntegerAttr>(conOp.getValue()))
          return iattr.getInt() > 0;
      return false;
    };

    auto *op1 = removeConvert(v1);
    auto *op2 = removeConvert(v2);
    if (!op1 || !op2)
      return false;
    if (auto addi = aiir::dyn_cast<aiir::arith::AddIOp>(op2))
      if ((addi.getLhs().getDefiningOp() == op1 &&
           isPositiveConstant(addi.getRhs())) ||
          (addi.getRhs().getDefiningOp() == op1 &&
           isPositiveConstant(addi.getLhs())))
        return true;
    if (auto subi = aiir::dyn_cast<aiir::arith::SubIOp>(op1))
      if (subi.getLhs().getDefiningOp() == op2 &&
          isPositiveConstant(subi.getRhs()))
        return true;
    return false;
  };

  for (std::remove_const_t<decltype(size)> i = 0; i < size; i += 3) {
    // If both are loop invariant, skip to the next triple.
    if (aiir::isa_and_nonnull<fir::UndefOp>(ldTriples[i + 1].getDefiningOp()) &&
        aiir::isa_and_nonnull<fir::UndefOp>(stTriples[i + 1].getDefiningOp())) {
      // Unless either is a vector index, then be conservative.
      if (aiir::isa_and_nonnull<fir::UndefOp>(ldTriples[i].getDefiningOp()) ||
          aiir::isa_and_nonnull<fir::UndefOp>(stTriples[i].getDefiningOp()))
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
/// If `optimize` is true, use the variable attributes to prove that
/// there is no conflict.
static bool conflictOnLoad(llvm::ArrayRef<aiir::Operation *> reach,
                           ArrayMergeStoreOp st, bool optimize) {
  aiir::Value load;
  aiir::Value addr = st.getMemref();
  const bool storeHasPointerType = hasPointerType(addr.getType());
  for (auto *op : reach)
    if (auto ld = aiir::dyn_cast<ArrayLoadOp>(op)) {
      aiir::Type ldTy = ld.getMemref().getType();
      auto globalOpName = aiir::OperationName(fir::GlobalOp::getOperationName(),
                                              ld.getContext());
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
      } else if (storeHasPointerType) {
        if (optimize && !hasPointerType(ldTy) &&
            !valueMayHaveFirAttributes(
                ld.getMemref(),
                {getTargetAttrName(),
                 fir::GlobalOp::getTargetAttrName(globalOpName).strref()}))
          continue;

        return true;
      } else if (hasPointerType(ldTy)) {
        if (optimize && !storeHasPointerType &&
            !valueMayHaveFirAttributes(
                addr,
                {getTargetAttrName(),
                 fir::GlobalOp::getTargetAttrName(globalOpName).strref()}))
          continue;

        return true;
      }
      // TODO: Check if types can also allow ruling out some cases. For now,
      // the fact that equivalences is using pointer attribute to enforce
      // aliasing is preventing any attempt to do so, and in general, it may
      // be wrong to use this if any of the types is a complex or a derived
      // for which it is possible to create a pointer to a part with a
      // different type than the whole, although this deserve some more
      // investigation because existing compiler behavior seem to diverge
      // here.
    }
  return false;
}

/// Is there an access vector conflict on the array being merged into? If the
/// access vectors diverge, then assume that there are potentially overlapping
/// loop-carried references.
static bool conflictOnMerge(llvm::ArrayRef<aiir::Operation *> mentions) {
  if (mentions.size() < 2)
    return false;
  llvm::SmallVector<aiir::Value> indices;
  LLVM_DEBUG(llvm::dbgs() << "check merge conflict on with " << mentions.size()
                          << " mentions on the list\n");
  bool valSeen = false;
  bool refSeen = false;
  for (auto *op : mentions) {
    llvm::SmallVector<aiir::Value> compareVector;
    if (auto u = aiir::dyn_cast<ArrayUpdateOp>(op)) {
      valSeen = true;
      if (indices.empty()) {
        indices = u.getIndices();
        continue;
      }
      compareVector = u.getIndices();
    } else if (auto f = aiir::dyn_cast<ArrayModifyOp>(op)) {
      valSeen = true;
      if (indices.empty()) {
        indices = f.getIndices();
        continue;
      }
      compareVector = f.getIndices();
    } else if (auto f = aiir::dyn_cast<ArrayFetchOp>(op)) {
      valSeen = true;
      if (indices.empty()) {
        indices = f.getIndices();
        continue;
      }
      compareVector = f.getIndices();
    } else if (auto f = aiir::dyn_cast<ArrayAccessOp>(op)) {
      refSeen = true;
      if (indices.empty()) {
        indices = f.getIndices();
        continue;
      }
      compareVector = f.getIndices();
    } else if (aiir::isa<ArrayAmendOp>(op)) {
      refSeen = true;
      continue;
    } else {
      aiir::emitError(op->getLoc(), "unexpected operation in analysis");
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
static bool conflictOnReference(llvm::ArrayRef<aiir::Operation *> mentions) {
  LLVM_DEBUG(llvm::dbgs() << "checking reference semantics " << mentions.size()
                          << '\n');
  if (mentions.size() < 3)
    return false;
  unsigned amendCount = 0;
  unsigned accessCount = 0;
  for (auto *op : mentions) {
    if (aiir::isa<ArrayAmendOp>(op) && ++amendCount > 1) {
      LLVM_DEBUG(llvm::dbgs() << "conflict: multiple amends of array value\n");
      return true;
    }
    if (aiir::isa<ArrayAccessOp>(op) && ++accessCount > 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "conflict: multiple accesses of array value\n");
      return true;
    }
    if (aiir::isa<ArrayFetchOp, ArrayUpdateOp, ArrayModifyOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "conflict: array value has both uses by-value and uses "
                    "by-reference. conservative assumption.\n");
      return true;
    }
  }
  return false;
}

static aiir::Operation *
amendingAccess(llvm::ArrayRef<aiir::Operation *> mentions) {
  for (auto *op : mentions)
    if (auto amend = aiir::dyn_cast<ArrayAmendOp>(op))
      return amend.getMemref().getDefiningOp();
  return {};
}

// Are any conflicts present? The conflicts detected here are described above.
static bool conflictDetected(llvm::ArrayRef<aiir::Operation *> reach,
                             llvm::ArrayRef<aiir::Operation *> mentions,
                             ArrayMergeStoreOp st, bool optimize) {
  return conflictOnLoad(reach, st, optimize) || conflictOnMerge(mentions);
}

// Assume that any call to a function that uses host-associations will be
// modifying the output array.
static bool
conservativeCallConflict(llvm::ArrayRef<aiir::Operation *> reaches) {
  return llvm::any_of(reaches, [](aiir::Operation *op) {
    if (auto call = aiir::dyn_cast<fir::CallOp>(op))
      if (auto callee = aiir::dyn_cast<aiir::SymbolRefAttr>(
              call.getCallableForCallee())) {
        auto module = op->getParentOfType<aiir::ModuleOp>();
        return isInternalProcedure(
            module.lookupSymbol<aiir::func::FuncOp>(callee));
      }
    return false;
  });
}

/// Constructor of the array copy analysis.
/// This performs the analysis and saves the intermediate results.
void ArrayCopyAnalysisBase::construct(aiir::Operation *topLevelOp) {
  topLevelOp->walk([&](Operation *op) {
    if (auto st = aiir::dyn_cast<fir::ArrayMergeStoreOp>(op)) {
      llvm::SmallVector<aiir::Operation *> values;
      ReachCollector::reachingValues(values, st.getSequence());
      bool callConflict = conservativeCallConflict(values);
      llvm::SmallVector<aiir::Operation *> mentions;
      arrayMentions(mentions,
                    aiir::cast<ArrayLoadOp>(st.getOriginal().getDefiningOp()));
      bool conflict = conflictDetected(values, mentions, st, optimizeConflicts);
      bool refConflict = conflictOnReference(mentions);
      if (callConflict || conflict || refConflict) {
        LLVM_DEBUG(llvm::dbgs()
                   << "CONFLICT: copies required for " << st << '\n'
                   << "   adding conflicts on: " << *op << " and "
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
    } else if (auto load = aiir::dyn_cast<ArrayLoadOp>(op)) {
      llvm::SmallVector<aiir::Operation *> mentions;
      arrayMentions(mentions, load);
      LLVM_DEBUG(llvm::dbgs() << "process load: " << load
                              << ", mentions: " << mentions.size() << '\n');
      for (auto *acc : mentions) {
        LLVM_DEBUG(llvm::dbgs() << " mention: " << *acc << '\n');
        if (aiir::isa<ArrayAccessOp, ArrayAmendOp, ArrayFetchOp, ArrayUpdateOp,
                      ArrayModifyOp>(acc)) {
          if (useMap.count(acc)) {
            aiir::emitError(
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
class ArrayLoadConversion : public aiir::OpRewritePattern<ArrayLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(ArrayLoadOp load,
                  aiir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "replace load " << load << " with undef.\n");
    rewriter.replaceOpWithNewOp<UndefOp>(load, load.getType());
    return aiir::success();
  }
};

class ArrayMergeStoreConversion
    : public aiir::OpRewritePattern<ArrayMergeStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(ArrayMergeStoreOp store,
                  aiir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "marking store " << store << " as dead.\n");
    rewriter.eraseOp(store);
    return aiir::success();
  }
};
} // namespace

static aiir::Type getEleTy(aiir::Type ty) {
  auto eleTy = unwrapSequenceType(unwrapPassByRefType(ty));
  // FIXME: keep ptr/heap/ref information.
  return ReferenceType::get(eleTy);
}

// This is an unsafe way to deduce this (won't be true in internal
// procedure or inside select-rank for assumed-size). Only here to satisfy
// legacy code until removed.
static bool isAssumedSize(llvm::SmallVectorImpl<aiir::Value> &extents) {
  if (extents.empty())
    return false;
  return llvm::isa_and_nonnull<fir::AssumedSizeExtentOp>(
      extents.back().getDefiningOp());
}

// Extract extents from the ShapeOp/ShapeShiftOp into the result vector.
static bool getAdjustedExtents(aiir::Location loc,
                               aiir::PatternRewriter &rewriter,
                               ArrayLoadOp arrLoad,
                               llvm::SmallVectorImpl<aiir::Value> &result,
                               aiir::Value shape) {
  bool copyUsingSlice = false;
  auto *shapeOp = shape.getDefiningOp();
  if (auto s = aiir::dyn_cast_or_null<ShapeOp>(shapeOp)) {
    auto e = s.getExtents();
    result.insert(result.end(), e.begin(), e.end());
  } else if (auto s = aiir::dyn_cast_or_null<ShapeShiftOp>(shapeOp)) {
    auto e = s.getExtents();
    result.insert(result.end(), e.begin(), e.end());
  } else {
    emitFatalError(loc, "not a fir.shape/fir.shape_shift op");
  }
  auto idxTy = rewriter.getIndexType();
  if (isAssumedSize(result)) {
    // Use slice information to compute the extent of the column.
    auto one = aiir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    aiir::Value size = one;
    if (aiir::Value sliceArg = arrLoad.getSlice()) {
      if (auto sliceOp =
              aiir::dyn_cast_or_null<SliceOp>(sliceArg.getDefiningOp())) {
        auto triples = sliceOp.getTriples();
        const std::size_t tripleSize = triples.size();
        auto module = arrLoad->getParentOfType<aiir::ModuleOp>();
        FirOpBuilder builder(rewriter, module);
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
static aiir::Value getOrReadExtentsAndShapeOp(
    aiir::Location loc, aiir::PatternRewriter &rewriter, ArrayLoadOp arrLoad,
    llvm::SmallVectorImpl<aiir::Value> &result, bool &copyUsingSlice) {
  assert(result.empty());
  if (arrLoad->hasAttr(fir::getOptionalAttrName()))
    fir::emitFatalError(
        loc, "shapes from array load of OPTIONAL arrays must not be used");
  if (auto boxTy = aiir::dyn_cast<BoxType>(arrLoad.getMemref().getType())) {
    auto rank =
        aiir::cast<SequenceType>(dyn_cast_ptrOrBoxEleTy(boxTy)).getDimension();
    auto idxTy = rewriter.getIndexType();
    for (decltype(rank) dim = 0; dim < rank; ++dim) {
      auto dimVal = aiir::arith::ConstantIndexOp::create(rewriter, loc, dim);
      auto dimInfo = BoxDimsOp::create(rewriter, loc, idxTy, idxTy, idxTy,
                                       arrLoad.getMemref(), dimVal);
      result.emplace_back(dimInfo.getResult(1));
    }
    if (!arrLoad.getShape()) {
      auto shapeType = ShapeType::get(rewriter.getContext(), rank);
      return ShapeOp::create(rewriter, loc, shapeType, result);
    }
    auto shiftOp = arrLoad.getShape().getDefiningOp<ShiftOp>();
    auto shapeShiftType = ShapeShiftType::get(rewriter.getContext(), rank);
    llvm::SmallVector<aiir::Value> shapeShiftOperands;
    for (auto [lb, extent] : llvm::zip(shiftOp.getOrigins(), result)) {
      shapeShiftOperands.push_back(lb);
      shapeShiftOperands.push_back(extent);
    }
    return ShapeShiftOp::create(rewriter, loc, shapeShiftType,
                                shapeShiftOperands);
  }
  copyUsingSlice =
      getAdjustedExtents(loc, rewriter, arrLoad, result, arrLoad.getShape());
  return arrLoad.getShape();
}

static aiir::Type toRefType(aiir::Type ty) {
  if (fir::isa_ref_type(ty))
    return ty;
  return fir::ReferenceType::get(ty);
}

static llvm::SmallVector<aiir::Value>
getTypeParamsIfRawData(aiir::Location loc, FirOpBuilder &builder,
                       ArrayLoadOp arrLoad, aiir::Type ty) {
  if (aiir::isa<BoxType>(ty))
    return {};
  return fir::factory::getTypeParams(loc, builder, arrLoad);
}

static aiir::Value genCoorOp(aiir::PatternRewriter &rewriter,
                             aiir::Location loc, aiir::Type eleTy,
                             aiir::Type resTy, aiir::Value alloc,
                             aiir::Value shape, aiir::Value slice,
                             aiir::ValueRange indices, ArrayLoadOp load,
                             bool skipOrig = false) {
  llvm::SmallVector<aiir::Value> originated;
  if (skipOrig)
    originated.assign(indices.begin(), indices.end());
  else
    originated = factory::originateIndices(loc, rewriter, alloc.getType(),
                                           shape, indices);
  auto seqTy = dyn_cast_ptrOrBoxEleTy(alloc.getType());
  assert(seqTy && aiir::isa<SequenceType>(seqTy));
  const auto dimension = aiir::cast<SequenceType>(seqTy).getDimension();
  auto module = load->getParentOfType<aiir::ModuleOp>();
  FirOpBuilder builder(rewriter, module);
  auto typeparams = getTypeParamsIfRawData(loc, builder, load, alloc.getType());
  aiir::Value result = ArrayCoorOp::create(
      rewriter, loc, eleTy, alloc, shape, slice,
      llvm::ArrayRef<aiir::Value>{originated}.take_front(dimension),
      typeparams);
  if (dimension < originated.size())
    result = fir::CoordinateOp::create(
        rewriter, loc, resTy, result,
        llvm::ArrayRef<aiir::Value>{originated}.drop_front(dimension));
  return result;
}

static aiir::Value getCharacterLen(aiir::Location loc, FirOpBuilder &builder,
                                   ArrayLoadOp load, CharacterType charTy) {
  auto charLenTy = builder.getCharacterLengthType();
  if (charTy.hasDynamicLen()) {
    if (aiir::isa<BoxType>(load.getMemref().getType())) {
      // The loaded array is an emboxed value. Get the CHARACTER length from
      // the box value.
      auto eleSzInBytes =
          BoxEleSizeOp::create(builder, loc, charLenTy, load.getMemref());
      auto kindSize =
          builder.getKindMap().getCharacterBitsize(charTy.getFKind());
      auto kindByteSize =
          builder.createIntegerConstant(loc, charLenTy, kindSize / 8);
      return aiir::arith::DivSIOp::create(builder, loc, eleSzInBytes,
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
void genArrayCopy(aiir::Location loc, aiir::PatternRewriter &rewriter,
                  aiir::Value dst, aiir::Value src, aiir::Value shapeOp,
                  aiir::Value sliceOp, ArrayLoadOp arrLoad) {
  auto insPt = rewriter.saveInsertionPoint();
  llvm::SmallVector<aiir::Value> indices;
  llvm::SmallVector<aiir::Value> extents;
  bool copyUsingSlice =
      getAdjustedExtents(loc, rewriter, arrLoad, extents, shapeOp);
  auto idxTy = rewriter.getIndexType();
  // Build loop nest from column to row.
  for (auto sh : llvm::reverse(extents)) {
    auto ubi = ConvertOp::create(rewriter, loc, idxTy, sh);
    auto zero = aiir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto one = aiir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto ub = aiir::arith::SubIOp::create(rewriter, loc, idxTy, ubi, one);
    auto loop = DoLoopOp::create(rewriter, loc, zero, ub, one);
    rewriter.setInsertionPointToStart(loop.getBody());
    indices.push_back(loop.getInductionVar());
  }
  // Reverse the indices so they are in column-major order.
  std::reverse(indices.begin(), indices.end());
  auto module = arrLoad->getParentOfType<aiir::ModuleOp>();
  FirOpBuilder builder(rewriter, module);
  auto fromAddr = ArrayCoorOp::create(
      rewriter, loc, getEleTy(src.getType()), src, shapeOp,
      CopyIn && copyUsingSlice ? sliceOp : aiir::Value{},
      factory::originateIndices(loc, rewriter, src.getType(), shapeOp, indices),
      getTypeParamsIfRawData(loc, builder, arrLoad, src.getType()));
  auto toAddr = ArrayCoorOp::create(
      rewriter, loc, getEleTy(dst.getType()), dst, shapeOp,
      !CopyIn && copyUsingSlice ? sliceOp : aiir::Value{},
      factory::originateIndices(loc, rewriter, dst.getType(), shapeOp, indices),
      getTypeParamsIfRawData(loc, builder, arrLoad, dst.getType()));
  auto eleTy = unwrapSequenceType(unwrapPassByRefType(dst.getType()));
  // Copy from (to) object to (from) temp copy of same object.
  if (auto charTy = aiir::dyn_cast<CharacterType>(eleTy)) {
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
static llvm::SmallVector<aiir::Value>
genArrayLoadTypeParameters(aiir::Location loc, aiir::PatternRewriter &rewriter,
                           ArrayLoadOp load) {
  if (load.getTypeparams().empty()) {
    auto eleTy =
        unwrapSequenceType(unwrapPassByRefType(load.getMemref().getType()));
    if (hasDynamicSize(eleTy)) {
      if (auto charTy = aiir::dyn_cast<CharacterType>(eleTy)) {
        assert(aiir::isa<BoxType>(load.getMemref().getType()));
        auto module = load->getParentOfType<aiir::ModuleOp>();
        FirOpBuilder builder(rewriter, module);
        return {getCharacterLen(loc, builder, load, charTy)};
      }
      TODO(loc, "unhandled dynamic type parameters");
    }
    return {};
  }
  return load.getTypeparams();
}

static llvm::SmallVector<aiir::Value>
findNonconstantExtents(aiir::Type memrefTy,
                       llvm::ArrayRef<aiir::Value> extents) {
  llvm::SmallVector<aiir::Value> nce;
  auto arrTy = unwrapPassByRefType(memrefTy);
  auto seqTy = aiir::cast<SequenceType>(arrTy);
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
static std::pair<aiir::Value,
                 std::function<void(aiir::PatternRewriter &rewriter)>>
allocateArrayTemp(aiir::Location loc, aiir::PatternRewriter &rewriter,
                  ArrayLoadOp load, llvm::ArrayRef<aiir::Value> extents,
                  aiir::Value shape) {
  aiir::Type baseType = load.getMemref().getType();
  llvm::SmallVector<aiir::Value> nonconstantExtents =
      findNonconstantExtents(baseType, extents);
  llvm::SmallVector<aiir::Value> typeParams =
      genArrayLoadTypeParameters(loc, rewriter, load);
  aiir::Value allocmem =
      AllocMemOp::create(rewriter, loc, dyn_cast_ptrOrBoxEleTy(baseType),
                         typeParams, nonconstantExtents);
  aiir::Type eleType =
      fir::unwrapSequenceType(fir::unwrapPassByRefType(baseType));
  if (fir::isRecordWithAllocatableMember(eleType)) {
    // The allocatable component descriptors need to be set to a clean
    // deallocated status before anything is done with them.
    aiir::Value box = fir::EmboxOp::create(
        rewriter, loc, fir::BoxType::get(allocmem.getType()), allocmem, shape,
        /*slice=*/aiir::Value{}, typeParams);
    auto module = load->getParentOfType<aiir::ModuleOp>();
    FirOpBuilder builder(rewriter, module);
    runtime::genDerivedTypeInitialize(builder, loc, box);
    // Any allocatable component that may have been allocated must be
    // deallocated during the clean-up.
    auto cleanup = [=](aiir::PatternRewriter &r) {
      FirOpBuilder builder(r, module);
      runtime::genDerivedTypeDestroy(builder, loc, box);
      FreeMemOp::create(r, loc, allocmem);
    };
    return {allocmem, cleanup};
  }
  auto cleanup = [=](aiir::PatternRewriter &r) {
    FreeMemOp::create(r, loc, allocmem);
  };
  return {allocmem, cleanup};
}

namespace {
/// Conversion of fir.array_update and fir.array_modify Ops.
/// If there is a conflict for the update, then we need to perform a
/// copy-in/copy-out to preserve the original values of the array. If there is
/// no conflict, then it is save to eschew making any copies.
template <typename ArrayOp>
class ArrayUpdateConversionBase : public aiir::OpRewritePattern<ArrayOp> {
public:
  // TODO: Implement copy/swap semantics?
  explicit ArrayUpdateConversionBase(aiir::AIIRContext *ctx,
                                     const ArrayCopyAnalysisBase &a,
                                     const OperationUseMapT &m)
      : aiir::OpRewritePattern<ArrayOp>{ctx}, analysis{a}, useMap{m} {}

  /// The array_access, \p access, is to be to a cloned copy due to a potential
  /// conflict. Uses copy-in/copy-out semantics and not copy/swap.
  aiir::Value referenceToClone(aiir::Location loc,
                               aiir::PatternRewriter &rewriter,
                               ArrayOp access) const {
    LLVM_DEBUG(llvm::dbgs()
               << "generating copy-in/copy-out loops for " << access << '\n');
    auto *op = access.getOperation();
    auto *loadOp = useMap.lookup(op);
    auto load = aiir::cast<ArrayLoadOp>(loadOp);
    auto eleTy = access.getType();
    rewriter.setInsertionPoint(loadOp);
    // Copy in.
    llvm::SmallVector<aiir::Value> extents;
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
    auto coor = genCoorOp(
        rewriter, loc, getEleTy(load.getType()), eleTy, allocmem, shapeOp,
        copyUsingSlice ? aiir::Value{} : load.getSlice(), access.getIndices(),
        load, access->hasAttr(factory::attrFortranArrayOffsets()));
    // Copy out.
    auto *storeOp = useMap.lookup(loadOp);
    auto store = aiir::cast<ArrayMergeStoreOp>(storeOp);
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
  std::pair<aiir::Value, aiir::Value>
  materializeAssignment(aiir::Location loc, aiir::PatternRewriter &rewriter,
                        ArrayOp update,
                        const std::function<void(aiir::Value)> &assignElement,
                        aiir::Type lhsEltRefType) const {
    auto *op = update.getOperation();
    auto *loadOp = useMap.lookup(op);
    auto load = aiir::cast<ArrayLoadOp>(loadOp);
    LLVM_DEBUG(llvm::outs() << "does " << load << " have a conflict?\n");
    if (analysis.hasPotentialConflict(loadOp)) {
      // If there is a conflict between the arrays, then we copy the lhs array
      // to a temporary, update the temporary, and copy the temporary back to
      // the lhs array. This yields Fortran's copy-in copy-out array semantics.
      LLVM_DEBUG(llvm::outs() << "Yes, conflict was found\n");
      rewriter.setInsertionPoint(loadOp);
      // Copy in.
      llvm::SmallVector<aiir::Value> extents;
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
          shapeOp, copyUsingSlice ? aiir::Value{} : load.getSlice(),
          update.getIndices(), load,
          update->hasAttr(factory::attrFortranArrayOffsets()));
      assignElement(coor);
      auto *storeOp = useMap.lookup(loadOp);
      auto store = aiir::cast<ArrayMergeStoreOp>(storeOp);
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
    auto coor =
        genCoorOp(rewriter, loc, coorTy, lhsEltRefType, load.getMemref(),
                  load.getShape(), load.getSlice(), update.getIndices(), load,
                  update->hasAttr(factory::attrFortranArrayOffsets()));
    assignElement(coor);
    return {coor, load.getResult()};
  }

protected:
  const ArrayCopyAnalysisBase &analysis;
  const OperationUseMapT &useMap;
};

class ArrayUpdateConversion : public ArrayUpdateConversionBase<ArrayUpdateOp> {
public:
  explicit ArrayUpdateConversion(aiir::AIIRContext *ctx,
                                 const ArrayCopyAnalysisBase &a,
                                 const OperationUseMapT &m)
      : ArrayUpdateConversionBase{ctx, a, m} {}

  llvm::LogicalResult
  matchAndRewrite(ArrayUpdateOp update,
                  aiir::PatternRewriter &rewriter) const override {
    auto loc = update.getLoc();
    auto assignElement = [&](aiir::Value coor) {
      auto input = update.getMerge();
      if (auto inEleTy = dyn_cast_ptrEleTy(input.getType())) {
        emitFatalError(loc, "array_update on references not supported");
      } else {
        fir::StoreOp::create(rewriter, loc, input, coor);
      }
    };
    auto lhsEltRefType = toRefType(update.getMerge().getType());
    auto [_, lhsLoadResult] = materializeAssignment(
        loc, rewriter, update, assignElement, lhsEltRefType);
    rewriter.replaceOp(update, lhsLoadResult);
    return aiir::success();
  }
};

class ArrayModifyConversion : public ArrayUpdateConversionBase<ArrayModifyOp> {
public:
  explicit ArrayModifyConversion(aiir::AIIRContext *ctx,
                                 const ArrayCopyAnalysisBase &a,
                                 const OperationUseMapT &m)
      : ArrayUpdateConversionBase{ctx, a, m} {}

  llvm::LogicalResult
  matchAndRewrite(ArrayModifyOp modify,
                  aiir::PatternRewriter &rewriter) const override {
    auto loc = modify.getLoc();
    auto assignElement = [](aiir::Value) {
      // Assignment already materialized by lowering using lhs element address.
    };
    auto lhsEltRefType = modify.getResult(0).getType();
    auto [lhsEltCoor, lhsLoadResult] = materializeAssignment(
        loc, rewriter, modify, assignElement, lhsEltRefType);
    rewriter.replaceOp(modify, aiir::ValueRange{lhsEltCoor, lhsLoadResult});
    return aiir::success();
  }
};

class ArrayFetchConversion : public aiir::OpRewritePattern<ArrayFetchOp> {
public:
  explicit ArrayFetchConversion(aiir::AIIRContext *ctx,
                                const OperationUseMapT &m)
      : OpRewritePattern{ctx}, useMap{m} {}

  llvm::LogicalResult
  matchAndRewrite(ArrayFetchOp fetch,
                  aiir::PatternRewriter &rewriter) const override {
    auto *op = fetch.getOperation();
    rewriter.setInsertionPoint(op);
    auto load = aiir::cast<ArrayLoadOp>(useMap.lookup(op));
    auto loc = fetch.getLoc();
    auto coor = genCoorOp(
        rewriter, loc, getEleTy(load.getType()), toRefType(fetch.getType()),
        load.getMemref(), load.getShape(), load.getSlice(), fetch.getIndices(),
        load, fetch->hasAttr(factory::attrFortranArrayOffsets()));
    if (isa_ref_type(fetch.getType()))
      rewriter.replaceOp(fetch, coor);
    else
      rewriter.replaceOpWithNewOp<fir::LoadOp>(fetch, coor);
    return aiir::success();
  }

private:
  const OperationUseMapT &useMap;
};

/// As array_access op is like an array_fetch op, except that it does not imply
/// a load op. (It operates in the reference domain.)
class ArrayAccessConversion : public ArrayUpdateConversionBase<ArrayAccessOp> {
public:
  explicit ArrayAccessConversion(aiir::AIIRContext *ctx,
                                 const ArrayCopyAnalysisBase &a,
                                 const OperationUseMapT &m)
      : ArrayUpdateConversionBase{ctx, a, m} {}

  llvm::LogicalResult
  matchAndRewrite(ArrayAccessOp access,
                  aiir::PatternRewriter &rewriter) const override {
    auto *op = access.getOperation();
    auto loc = access.getLoc();
    if (analysis.inAmendAccessSet(op)) {
      // This array_access is associated with an array_amend and there is a
      // conflict. Make a copy to store into.
      auto result = referenceToClone(loc, rewriter, access);
      rewriter.replaceOp(access, result);
      return aiir::success();
    }
    rewriter.setInsertionPoint(op);
    auto load = aiir::cast<ArrayLoadOp>(useMap.lookup(op));
    auto coor = genCoorOp(
        rewriter, loc, getEleTy(load.getType()), toRefType(access.getType()),
        load.getMemref(), load.getShape(), load.getSlice(), access.getIndices(),
        load, access->hasAttr(factory::attrFortranArrayOffsets()));
    rewriter.replaceOp(access, coor);
    return aiir::success();
  }
};

/// An array_amend op is a marker to record which array access is being used to
/// update an array value. After this pass runs, an array_amend has no
/// semantics. We rewrite these to undefined values here to remove them while
/// preserving SSA form.
class ArrayAmendConversion : public aiir::OpRewritePattern<ArrayAmendOp> {
public:
  explicit ArrayAmendConversion(aiir::AIIRContext *ctx)
      : OpRewritePattern{ctx} {}

  llvm::LogicalResult
  matchAndRewrite(ArrayAmendOp amend,
                  aiir::PatternRewriter &rewriter) const override {
    auto *op = amend.getOperation();
    rewriter.setInsertionPoint(op);
    auto loc = amend.getLoc();
    auto undef = UndefOp::create(rewriter, loc, amend.getType());
    rewriter.replaceOp(amend, undef.getResult());
    return aiir::success();
  }
};

class ArrayValueCopyConverter
    : public fir::impl::ArrayValueCopyBase<ArrayValueCopyConverter> {
public:
  ArrayValueCopyConverter() = default;
  ArrayValueCopyConverter(const fir::ArrayValueCopyOptions &options)
      : Base(options) {}

  void runOnOperation() override {
    auto func = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "\n\narray-value-copy pass on function '"
                            << func.getName() << "'\n");
    auto *context = &getContext();

    // Perform the conflict analysis.
    const ArrayCopyAnalysisBase *analysis;
    if (optimizeConflicts)
      analysis = &getAnalysis<ArrayCopyAnalysisOptimized>();
    else
      analysis = &getAnalysis<ArrayCopyAnalysis>();

    const auto &useMap = analysis->getUseMap();

    aiir::RewritePatternSet patterns1(context);
    patterns1.insert<ArrayFetchConversion>(context, useMap);
    patterns1.insert<ArrayUpdateConversion>(context, *analysis, useMap);
    patterns1.insert<ArrayModifyConversion>(context, *analysis, useMap);
    patterns1.insert<ArrayAccessConversion>(context, *analysis, useMap);
    patterns1.insert<ArrayAmendConversion>(context);
    aiir::ConversionTarget target(*context);
    target
        .addLegalDialect<FIROpsDialect, aiir::scf::SCFDialect,
                         aiir::arith::ArithDialect, aiir::func::FuncDialect>();
    target.addIllegalOp<ArrayAccessOp, ArrayAmendOp, ArrayFetchOp,
                        ArrayUpdateOp, ArrayModifyOp>();
    // Rewrite the array fetch and array update ops.
    if (aiir::failed(
            aiir::applyPartialConversion(func, target, std::move(patterns1)))) {
      aiir::emitError(aiir::UnknownLoc::get(context),
                      "failure in array-value-copy pass, phase 1");
      signalPassFailure();
    }

    aiir::RewritePatternSet patterns2(context);
    patterns2.insert<ArrayLoadConversion>(context);
    patterns2.insert<ArrayMergeStoreConversion>(context);
    target.addIllegalOp<ArrayLoadOp, ArrayMergeStoreOp>();
    if (aiir::failed(
            aiir::applyPartialConversion(func, target, std::move(patterns2)))) {
      aiir::emitError(aiir::UnknownLoc::get(context),
                      "failure in array-value-copy pass, phase 2");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<aiir::Pass>
fir::createArrayValueCopyPass(fir::ArrayValueCopyOptions options) {
  return std::make_unique<ArrayValueCopyConverter>(options);
}
