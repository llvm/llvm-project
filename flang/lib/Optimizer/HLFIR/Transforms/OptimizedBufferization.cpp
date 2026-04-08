//===- OptimizedBufferization.cpp - special cases for bufferization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// In some special cases we can bufferize hlfir expressions in a more optimal
// way so as to avoid creating temporaries. This pass handles these. It should
// be run before the catch-all bufferization pass.
//
// This requires constant subexpression elimination to have already been run.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Analysis/ArraySectionAnalyzer.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Transforms/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iterator>
#include <memory>
#include <mlir/Analysis/AliasAnalysis.h>
#include <optional>

namespace hlfir {
#define GEN_PASS_DEF_OPTIMIZEDBUFFERIZATION
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

#define DEBUG_TYPE "opt-bufferization"

namespace {

/// This transformation should match in place modification of arrays.
/// It should match code of the form
/// %array = some.operation // array has shape %shape
/// %expr = hlfir.elemental %shape : [...] {
/// bb0(%arg0: index)
///   %0 = hlfir.designate %array(%arg0)
///   [...] // no other reads or writes to %array
///   hlfir.yield_element %element
/// }
/// hlfir.assign %expr to %array
/// hlfir.destroy %expr
///
/// Or
///
/// %read_array = some.operation // shape %shape
/// %expr = hlfir.elemental %shape : [...] {
/// bb0(%arg0: index)
///   %0 = hlfir.designate %read_array(%arg0)
///   [...]
///   hlfir.yield_element %element
/// }
/// %write_array = some.operation // with shape %shape
/// [...] // operations which don't effect write_array
/// hlfir.assign %expr to %write_array
/// hlfir.destroy %expr
///
/// In these cases, it is safe to turn the elemental into a do loop and modify
/// elements of %array in place without creating an extra temporary for the
/// elemental. We must check that there are no reads from the array at indexes
/// which might conflict with the assignment or any writes. For now we will keep
/// that strict and say that all reads must be at the elemental index (it is
/// probably safe to read from higher indices if lowering to an ordered loop).
class ElementalAssignBufferization
    : public mlir::OpRewritePattern<hlfir::ElementalOp> {
private:
  struct MatchInfo {
    mlir::Value array;
    hlfir::AssignOp assign;
    hlfir::DestroyOp destroy;
  };
  /// determines if the transformation can be applied to this elemental
  static std::optional<MatchInfo> findMatch(hlfir::ElementalOp elemental);

public:
  using mlir::OpRewritePattern<hlfir::ElementalOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::ElementalOp elemental,
                  mlir::PatternRewriter &rewriter) const override;
};

/// recursively collect all effects between start and end (including start, not
/// including end) start must properly dominate end, start and end must be in
/// the same block. If any operations with unknown effects are found,
/// std::nullopt is returned
static std::optional<mlir::SmallVector<mlir::MemoryEffects::EffectInstance>>
getEffectsBetween(mlir::Operation *start, mlir::Operation *end) {
  mlir::SmallVector<mlir::MemoryEffects::EffectInstance> ret;
  if (start == end)
    return ret;
  assert(start->getBlock() && end->getBlock() && "TODO: block arguments");
  assert(start->getBlock() == end->getBlock());
  assert(mlir::DominanceInfo{}.properlyDominates(start, end));

  mlir::Operation *nextOp = start;
  while (nextOp && nextOp != end) {
    std::optional<mlir::SmallVector<mlir::MemoryEffects::EffectInstance>>
        effects = mlir::getEffectsRecursively(nextOp);
    if (!effects)
      return std::nullopt;
    ret.append(*effects);
    nextOp = nextOp->getNextNode();
  }
  return ret;
}

/// If effect is a read or write on val, return whether it aliases.
/// Otherwise return mlir::AliasResult::NoAlias
static mlir::AliasResult
containsReadOrWriteEffectOn(const mlir::MemoryEffects::EffectInstance &effect,
                            mlir::Value val) {
  fir::AliasAnalysis aliasAnalysis;

  if (mlir::isa<mlir::MemoryEffects::Read, mlir::MemoryEffects::Write>(
          effect.getEffect())) {
    mlir::Value accessedVal = effect.getValue();
    if (mlir::isa<fir::DebuggingResource>(effect.getResource()))
      return mlir::AliasResult::NoAlias;
    if (!accessedVal)
      return mlir::AliasResult::MayAlias;
    if (accessedVal == val)
      return mlir::AliasResult::MustAlias;

    // if the accessed value might alias val
    mlir::AliasResult res = aliasAnalysis.alias(val, accessedVal);
    if (!res.isNo())
      return res;

    // FIXME: alias analysis of fir.load
    // follow this common pattern:
    // %ref = hlfir.designate %array(%index)
    // %val = fir.load $ref
    if (auto designate = accessedVal.getDefiningOp<hlfir::DesignateOp>()) {
      if (designate.getMemref() == val)
        return mlir::AliasResult::MustAlias;

      // if the designate is into an array that might alias val
      res = aliasAnalysis.alias(val, designate.getMemref());
      if (!res.isNo())
        return res;
    }
  }
  return mlir::AliasResult::NoAlias;
}

std::optional<ElementalAssignBufferization::MatchInfo>
ElementalAssignBufferization::findMatch(hlfir::ElementalOp elemental) {
  mlir::Operation::user_range users = elemental->getUsers();
  // the only uses of the elemental should be the assignment and the destroy
  if (std::distance(users.begin(), users.end()) != 2) {
    LLVM_DEBUG(llvm::dbgs() << "Too many uses of the elemental\n");
    return std::nullopt;
  }

  // If the ElementalOp must produce a temporary (e.g. for
  // finalization purposes), then we cannot inline it.
  if (hlfir::elementalOpMustProduceTemp(elemental)) {
    LLVM_DEBUG(llvm::dbgs() << "ElementalOp must produce a temp\n");
    return std::nullopt;
  }

  MatchInfo match;
  for (mlir::Operation *user : users)
    mlir::TypeSwitch<mlir::Operation *, void>(user)
        .Case([&](hlfir::AssignOp op) { match.assign = op; })
        .Case([&](hlfir::DestroyOp op) { match.destroy = op; });

  if (!match.assign || !match.destroy) {
    LLVM_DEBUG(llvm::dbgs() << "Couldn't find assign or destroy\n");
    return std::nullopt;
  }

  // the array is what the elemental is assigned into
  // TODO: this could be extended to also allow hlfir.expr by first bufferizing
  // the incoming expression
  match.array = match.assign.getLhs();
  mlir::Type arrayType = mlir::dyn_cast<fir::SequenceType>(
      fir::unwrapPassByRefType(match.array.getType()));
  if (!arrayType) {
    LLVM_DEBUG(llvm::dbgs() << "AssignOp's result is not an array\n");
    return std::nullopt;
  }

  // require that the array elements are trivial
  // TODO: this is just to make the pass easier to think about. Not an inherent
  // limitation
  mlir::Type eleTy = hlfir::getFortranElementType(arrayType);
  if (!fir::isa_trivial(eleTy)) {
    LLVM_DEBUG(llvm::dbgs() << "AssignOp's data type is not trivial\n");
    return std::nullopt;
  }

  // The array must have the same shape as the elemental.
  //
  // f2018 10.2.1.2 (3) requires the lhs and rhs of an assignment to be
  // conformable unless the lhs is an allocatable array. In HLFIR we can
  // see this from the presence or absence of the realloc attribute on
  // hlfir.assign. If it is not a realloc assignment, we can trust that
  // the shapes do conform.
  //
  // TODO: the lhs's shape is dynamic, so it is hard to prove that
  // there is no reallocation of the lhs due to the assignment.
  // We can probably try generating multiple versions of the code
  // with checking for the shape match, length parameters match, etc.
  if (match.assign.isAllocatableAssignment()) {
    LLVM_DEBUG(llvm::dbgs() << "AssignOp may involve (re)allocation of LHS\n");
    return std::nullopt;
  }

  // the transformation wants to apply the elemental in a do-loop at the
  // hlfir.assign, check there are no effects which make this unsafe

  // keep track of any values written to in the elemental, as these can't be
  // read from or written to between the elemental and the assignment
  mlir::SmallVector<mlir::Value, 1> notToBeAccessedBeforeAssign;
  // likewise, values read in the elemental cannot be written to between the
  // elemental and the assign
  mlir::SmallVector<mlir::Value, 1> notToBeWrittenBeforeAssign;

  // 1) side effects in the elemental body - it isn't sufficient to just look
  // for ordered elementals because we also cannot support out of order reads
  std::optional<mlir::SmallVector<mlir::MemoryEffects::EffectInstance>>
      effects = getEffectsBetween(&elemental.getBody()->front(),
                                  elemental.getBody()->getTerminator());
  if (!effects) {
    LLVM_DEBUG(llvm::dbgs()
               << "operation with unknown effects inside elemental\n");
    return std::nullopt;
  }
  for (const mlir::MemoryEffects::EffectInstance &effect : *effects) {
    mlir::AliasResult res = containsReadOrWriteEffectOn(effect, match.array);
    if (res.isNo()) {
      if (effect.getValue()) {
        if (mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect()))
          notToBeAccessedBeforeAssign.push_back(effect.getValue());
        else if (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()))
          notToBeWrittenBeforeAssign.push_back(effect.getValue());
      }

      // this is safe in the elemental
      continue;
    }

    // don't allow any aliasing writes in the elemental
    if (mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
      LLVM_DEBUG(llvm::dbgs() << "write inside the elemental body\n");
      return std::nullopt;
    }

    if (effect.getValue() == nullptr) {
      LLVM_DEBUG(llvm::dbgs()
                 << "side-effect with no value, cannot analyze further\n");
      return std::nullopt;
    }

    // allow if and only if the reads are from the elemental indices, in order
    // => each iteration doesn't read values written by other iterations
    // don't allow reads from a different value which may alias: fir alias
    // analysis isn't precise enough to tell us if two aliasing arrays overlap
    // exactly or only partially. If they overlap partially, a designate at the
    // elemental indices could be accessing different elements: e.g. we could
    // designate two slices of the same array at different start indexes. These
    // two MustAlias but index 1 of one array isn't the same element as index 1
    // of the other array.
    if (!res.isPartial()) {
      if (auto designate =
              effect.getValue().getDefiningOp<hlfir::DesignateOp>()) {
        fir::ArraySectionAnalyzer::SlicesOverlapKind overlap =
            fir::ArraySectionAnalyzer::analyze(match.array,
                                               designate.getMemref());
        if (overlap ==
            fir::ArraySectionAnalyzer::SlicesOverlapKind::DefinitelyDisjoint)
          continue;

        if (overlap == fir::ArraySectionAnalyzer::SlicesOverlapKind::Unknown) {
          LLVM_DEBUG(llvm::dbgs() << "possible read conflict: " << designate
                                  << " at " << elemental.getLoc() << "\n");
          return std::nullopt;
        }
        if (fir::ArraySectionAnalyzer::isDesignatingArrayInOrder(designate,
                                                                 elemental))
          continue;

        LLVM_DEBUG(llvm::dbgs() << "possible read conflict: " << designate
                                << " at " << elemental.getLoc() << "\n");
        return std::nullopt;
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "disallowed side-effect: " << effect.getValue()
                            << " for " << elemental.getLoc() << "\n");
    return std::nullopt;
  }

  // 2) look for conflicting effects between the elemental and the assignment
  effects = getEffectsBetween(elemental->getNextNode(), match.assign);
  if (!effects) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "operation with unknown effects between elemental and assign\n");
    return std::nullopt;
  }
  for (const mlir::MemoryEffects::EffectInstance &effect : *effects) {
    // not safe to access anything written in the elemental as this write
    // will be moved to the assignment
    for (mlir::Value val : notToBeAccessedBeforeAssign) {
      mlir::AliasResult res = containsReadOrWriteEffectOn(effect, val);
      if (!res.isNo()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "disallowed side-effect: " << effect.getValue() << " for "
                   << elemental.getLoc() << "\n");
        return std::nullopt;
      }
    }
    // Anything that is read inside the elemental can only be safely read
    // between the elemental and the assignment.
    for (mlir::Value val : notToBeWrittenBeforeAssign) {
      mlir::AliasResult res = containsReadOrWriteEffectOn(effect, val);
      if (!res.isNo() &&
          !mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "disallowed non-read side-effect: " << effect.getValue()
                   << " for " << elemental.getLoc() << "\n");
        return std::nullopt;
      }
    }
  }

  return match;
}

llvm::LogicalResult ElementalAssignBufferization::matchAndRewrite(
    hlfir::ElementalOp elemental, mlir::PatternRewriter &rewriter) const {
  std::optional<MatchInfo> match = findMatch(elemental);
  if (!match)
    return rewriter.notifyMatchFailure(
        elemental, "cannot prove safety of ElementalAssignBufferization");

  mlir::Location loc = elemental->getLoc();
  fir::FirOpBuilder builder(rewriter, elemental.getOperation());
  auto rhsExtents = hlfir::getIndexExtents(loc, builder, elemental.getShape());

  // create the loop at the assignment
  builder.setInsertionPoint(match->assign);
  hlfir::Entity lhs{match->array};
  lhs = hlfir::derefPointersAndAllocatables(loc, builder, lhs);
  mlir::Value lhsShape = hlfir::genShape(loc, builder, lhs);
  llvm::SmallVector<mlir::Value> lhsExtents =
      hlfir::getIndexExtents(loc, builder, lhsShape);
  llvm::SmallVector<mlir::Value> extents =
      fir::factory::deduceOptimalExtents(rhsExtents, lhsExtents);

  // Generate a loop nest looping around the hlfir.elemental shape and clone
  // hlfir.elemental region inside the inner loop
  hlfir::LoopNest loopNest =
      hlfir::genLoopNest(loc, builder, extents, !elemental.isOrdered(),
                         flangomp::shouldUseWorkshareLowering(elemental));
  builder.setInsertionPointToStart(loopNest.body);
  auto yield = hlfir::inlineElementalOp(loc, builder, elemental,
                                        loopNest.oneBasedIndices);
  hlfir::Entity elementValue{yield.getElementValue()};
  rewriter.eraseOp(yield);

  // Assign the element value to the array element for this iteration.
  auto arrayElement =
      hlfir::getElementAt(loc, builder, lhs, loopNest.oneBasedIndices);
  auto newAssign = hlfir::AssignOp::create(
      builder, loc, elementValue, arrayElement, /*realloc=*/false,
      /*keep_lhs_length_if_realloc=*/false, match->assign.getTemporaryLhs());
  if (auto accessGroups =
          match->assign.getOperation()->getAttrOfType<mlir::ArrayAttr>(
              fir::getAccessGroupsAttrName()))
    newAssign->setAttr(fir::getAccessGroupsAttrName(), accessGroups);

  rewriter.eraseOp(match->assign);
  rewriter.eraseOp(match->destroy);
  rewriter.eraseOp(elemental);
  return mlir::success();
}

/// Expand hlfir.assign of a scalar RHS to array LHS into a loop nest
/// of element-by-element assignments:
///   hlfir.assign %cst to %0 : f32, !fir.ref<!fir.array<6x6xf32>>
/// into:
///   fir.do_loop %arg0 = %c1 to %c6 step %c1 unordered {
///     fir.do_loop %arg1 = %c1 to %c6 step %c1 unordered {
///       %1 = hlfir.designate %0 (%arg1, %arg0)  :
///       (!fir.ref<!fir.array<6x6xf32>>, index, index) -> !fir.ref<f32>
///       hlfir.assign %cst to %1 : f32, !fir.ref<f32>
///     }
///   }
class BroadcastAssignBufferization
    : public mlir::OpRewritePattern<hlfir::AssignOp> {
private:
public:
  using mlir::OpRewritePattern<hlfir::AssignOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::AssignOp assign,
                  mlir::PatternRewriter &rewriter) const override;
};

llvm::LogicalResult BroadcastAssignBufferization::matchAndRewrite(
    hlfir::AssignOp assign, mlir::PatternRewriter &rewriter) const {
  // Since RHS is a scalar and LHS is an array, LHS must be allocated
  // in a conforming Fortran program, and LHS cannot be reallocated
  // as a result of the assignment. So we can ignore isAllocatableAssignment
  // and do the transformation always.
  mlir::Value rhs = assign.getRhs();
  if (!fir::isa_trivial(rhs.getType()))
    return rewriter.notifyMatchFailure(
        assign, "AssignOp's RHS is not a trivial scalar");

  hlfir::Entity lhs{assign.getLhs()};
  if (!lhs.isArray())
    return rewriter.notifyMatchFailure(assign,
                                       "AssignOp's LHS is not an array");

  mlir::Type eleTy = lhs.getFortranElementType();
  if (!fir::isa_trivial(eleTy))
    return rewriter.notifyMatchFailure(
        assign, "AssignOp's LHS data type is not trivial");

  mlir::Location loc = assign->getLoc();
  fir::FirOpBuilder builder(rewriter, assign.getOperation());
  builder.setInsertionPoint(assign);
  lhs = hlfir::derefPointersAndAllocatables(loc, builder, lhs);
  mlir::Value shape = hlfir::genShape(loc, builder, lhs);
  llvm::SmallVector<mlir::Value> extents =
      hlfir::getIndexExtents(loc, builder, shape);

  mlir::ArrayAttr accessGroups;
  if (auto attrs = assign.getOperation()->getAttrOfType<mlir::ArrayAttr>(
          fir::getAccessGroupsAttrName()))
    accessGroups = attrs;

  if (lhs.isSimplyContiguous() && extents.size() > 1) {
    // Flatten the array to use a single assign loop, that can be better
    // optimized.
    mlir::Value n = extents[0];
    for (size_t i = 1; i < extents.size(); ++i)
      n = mlir::arith::MulIOp::create(builder, loc, n, extents[i]);
    llvm::SmallVector<mlir::Value> flatExtents = {n};

    mlir::Type flatArrayType;
    mlir::Value flatArray = lhs.getBase();
    if (mlir::isa<fir::BoxType>(lhs.getType())) {
      shape = builder.genShape(loc, flatExtents);
      flatArrayType = fir::BoxType::get(fir::SequenceType::get(eleTy, 1));
      flatArray = fir::ReboxOp::create(builder, loc, flatArrayType, flatArray,
                                       shape, /*slice=*/mlir::Value{});
    } else {
      // Array references must have fixed shape, when used in assignments.
      auto seqTy =
          mlir::cast<fir::SequenceType>(fir::unwrapRefType(lhs.getType()));
      llvm::ArrayRef<int64_t> fixedShape = seqTy.getShape();
      int64_t flatExtent = 1;
      for (int64_t extent : fixedShape)
        flatExtent *= extent;
      flatArrayType =
          fir::ReferenceType::get(fir::SequenceType::get({flatExtent}, eleTy));
      flatArray = builder.createConvert(loc, flatArrayType, flatArray);
    }

    hlfir::LoopNest loopNest =
        hlfir::genLoopNest(loc, builder, flatExtents, /*isUnordered=*/true,
                           flangomp::shouldUseWorkshareLowering(assign));
    builder.setInsertionPointToStart(loopNest.body);

    mlir::Value arrayElement =
        hlfir::DesignateOp::create(builder, loc, fir::ReferenceType::get(eleTy),
                                   flatArray, loopNest.oneBasedIndices);
    auto newAssign = hlfir::AssignOp::create(builder, loc, rhs, arrayElement);
    if (accessGroups)
      newAssign->setAttr(fir::getAccessGroupsAttrName(), accessGroups);
  } else {
    hlfir::LoopNest loopNest =
        hlfir::genLoopNest(loc, builder, extents, /*isUnordered=*/true,
                           flangomp::shouldUseWorkshareLowering(assign));
    builder.setInsertionPointToStart(loopNest.body);
    auto arrayElement =
        hlfir::getElementAt(loc, builder, lhs, loopNest.oneBasedIndices);
    auto newAssign = hlfir::AssignOp::create(builder, loc, rhs, arrayElement);
    if (accessGroups)
      newAssign->setAttr(fir::getAccessGroupsAttrName(), accessGroups);
  }

  rewriter.eraseOp(assign);
  return mlir::success();
}

class EvaluateIntoMemoryAssignBufferization
    : public mlir::OpRewritePattern<hlfir::EvaluateInMemoryOp> {

public:
  using mlir::OpRewritePattern<hlfir::EvaluateInMemoryOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::EvaluateInMemoryOp,
                  mlir::PatternRewriter &rewriter) const override;
};

static llvm::LogicalResult
tryUsingAssignLhsDirectly(hlfir::EvaluateInMemoryOp evalInMem,
                          mlir::PatternRewriter &rewriter) {
  mlir::Location loc = evalInMem.getLoc();
  hlfir::DestroyOp destroy;
  hlfir::AssignOp assign;
  for (auto user : llvm::enumerate(evalInMem->getUsers())) {
    if (user.index() > 2)
      return mlir::failure();
    mlir::TypeSwitch<mlir::Operation *, void>(user.value())
        .Case([&](hlfir::AssignOp op) { assign = op; })
        .Case([&](hlfir::DestroyOp op) { destroy = op; });
  }
  if (!assign || !destroy || destroy.mustFinalizeExpr() ||
      assign.isAllocatableAssignment())
    return mlir::failure();

  hlfir::Entity lhs{assign.getLhs()};
  // EvaluateInMemoryOp memory is contiguous, so in general, it can only be
  // replace by the LHS if the LHS is contiguous.
  if (!lhs.isSimplyContiguous())
    return mlir::failure();
  // Character assignment may involves truncation/padding, so the LHS
  // cannot be used to evaluate RHS in place without proving the LHS and
  // RHS lengths are the same.
  if (lhs.isCharacter())
    return mlir::failure();
  fir::AliasAnalysis aliasAnalysis;
  // The region must not read or write the LHS.
  // Note that getModRef is used instead of mlir::MemoryEffects because
  // EvaluateInMemoryOp is typically expected to hold fir.calls and that
  // Fortran calls cannot be modeled in a useful way with mlir::MemoryEffects:
  // it is hard/impossible to list all the read/written SSA values in a call,
  // but it is often possible to tell that an SSA value cannot be accessed,
  // hence getModRef is needed here and below. Also note that getModRef uses
  // mlir::MemoryEffects for operations that do not have special handling in
  // getModRef.
  if (aliasAnalysis.getModRef(evalInMem.getBody(), lhs).isModOrRef())
    return mlir::failure();
  // Any variables affected between the hlfir.evalInMem and assignment must not
  // be read or written inside the region since it will be moved at the
  // assignment insertion point.
  auto effects = getEffectsBetween(evalInMem->getNextNode(), assign);
  if (!effects) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "operation with unknown effects between eval_in_mem and assign\n");
    return mlir::failure();
  }
  for (const mlir::MemoryEffects::EffectInstance &effect : *effects) {
    mlir::Value affected = effect.getValue();
    if (!affected ||
        aliasAnalysis.getModRef(evalInMem.getBody(), affected).isModOrRef())
      return mlir::failure();
  }

  rewriter.setInsertionPoint(assign);
  fir::FirOpBuilder builder(rewriter, evalInMem.getOperation());
  mlir::Value rawLhs = hlfir::genVariableRawAddress(loc, builder, lhs);
  hlfir::computeEvaluateOpIn(loc, builder, evalInMem, rawLhs);
  rewriter.eraseOp(assign);
  rewriter.eraseOp(destroy);
  rewriter.eraseOp(evalInMem);
  return mlir::success();
}

llvm::LogicalResult EvaluateIntoMemoryAssignBufferization::matchAndRewrite(
    hlfir::EvaluateInMemoryOp evalInMem,
    mlir::PatternRewriter &rewriter) const {
  if (mlir::succeeded(tryUsingAssignLhsDirectly(evalInMem, rewriter)))
    return mlir::success();
  // Rewrite to temp + as_expr here so that the assign + as_expr pattern can
  // kick-in for simple types and at least implement the assignment inline
  // instead of call Assign runtime.
  fir::FirOpBuilder builder(rewriter, evalInMem.getOperation());
  mlir::Location loc = evalInMem.getLoc();
  auto [temp, isHeapAllocated] = hlfir::computeEvaluateOpInNewTemp(
      loc, builder, evalInMem, evalInMem.getShape(), evalInMem.getTypeparams());
  rewriter.replaceOpWithNewOp<hlfir::AsExprOp>(
      evalInMem, temp, /*mustFree=*/builder.createBool(loc, isHeapAllocated));
  return mlir::success();
}

class OptimizedBufferizationPass
    : public hlfir::impl::OptimizedBufferizationBase<
          OptimizedBufferizationPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();

    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);

    mlir::RewritePatternSet patterns(context);
    // TODO: right now the patterns are non-conflicting,
    // but it might be better to run this pass on hlfir.assign
    // operations and decide which transformation to apply
    // at one place (e.g. we may use some heuristics and
    // choose different optimization strategies).
    // This requires small code reordering in ElementalAssignBufferization.
    patterns.insert<ElementalAssignBufferization>(context);
    patterns.insert<BroadcastAssignBufferization>(context);
    patterns.insert<EvaluateIntoMemoryAssignBufferization>(context);

    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in HLFIR optimized bufferization");
      signalPassFailure();
    }
  }
};
} // namespace
