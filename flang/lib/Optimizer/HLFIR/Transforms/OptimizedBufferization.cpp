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
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
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

  mlir::LogicalResult
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

// Returns true if the given array references represent identical
// or completely disjoint array slices. The callers may use this
// method when the alias analysis reports an alias of some kind,
// so that we can run Fortran specific analysis on the array slices
// to see if they are identical or disjoint. Note that the alias
// analysis are not able to give such an answer about the references.
static bool areIdenticalOrDisjointSlices(mlir::Value ref1, mlir::Value ref2) {
  if (ref1 == ref2)
    return true;

  auto des1 = ref1.getDefiningOp<hlfir::DesignateOp>();
  auto des2 = ref2.getDefiningOp<hlfir::DesignateOp>();
  // We only support a pair of designators right now.
  if (!des1 || !des2)
    return false;

  if (des1.getMemref() != des2.getMemref()) {
    // If the bases are different, then there is unknown overlap.
    LLVM_DEBUG(llvm::dbgs() << "No identical base for:\n"
                            << des1 << "and:\n"
                            << des2 << "\n");
    return false;
  }

  // Require all components of the designators to be the same.
  // It might be too strict, e.g. we may probably allow for
  // different type parameters.
  if (des1.getComponent() != des2.getComponent() ||
      des1.getComponentShape() != des2.getComponentShape() ||
      des1.getSubstring() != des2.getSubstring() ||
      des1.getComplexPart() != des2.getComplexPart() ||
      des1.getTypeparams() != des2.getTypeparams()) {
    LLVM_DEBUG(llvm::dbgs() << "Different designator specs for:\n"
                            << des1 << "and:\n"
                            << des2 << "\n");
    return false;
  }

  if (des1.getIsTriplet() != des2.getIsTriplet()) {
    LLVM_DEBUG(llvm::dbgs() << "Different sections for:\n"
                            << des1 << "and:\n"
                            << des2 << "\n");
    return false;
  }

  // Analyze the subscripts.
  // For example:
  //   hlfir.designate %6#0 (%c2:%c7999:%c1, %c1:%c120:%c1, %0)  shape %9
  //   hlfir.designate %6#0 (%c2:%c7999:%c1, %c1:%c120:%c1, %1)  shape %9
  //
  // If all the triplets (section speficiers) are the same, then
  // we do not care if %0 is equal to %1 - the slices are either
  // identical or completely disjoint.
  auto des1It = des1.getIndices().begin();
  auto des2It = des2.getIndices().begin();
  bool identicalTriplets = true;
  for (bool isTriplet : des1.getIsTriplet()) {
    if (isTriplet) {
      for (int i = 0; i < 3; ++i)
        if (*des1It++ != *des2It++) {
          LLVM_DEBUG(llvm::dbgs() << "Triplet mismatch for:\n"
                                  << des1 << "and:\n"
                                  << des2 << "\n");
          identicalTriplets = false;
          break;
        }
    } else {
      ++des1It;
      ++des2It;
    }
  }
  if (identicalTriplets)
    return true;

  // See if we can prove that any of the triplets do not overlap.
  // This is mostly a Polyhedron/nf performance hack that looks for
  // particular relations between the lower and upper bounds
  // of the array sections, e.g. for any positive constant C:
  //   X:Y does not overlap with (Y+C):Z
  //   X:Y does not overlap with Z:(X-C)
  auto displacedByConstant = [](mlir::Value v1, mlir::Value v2) {
    auto removeConvert = [](mlir::Value v) -> mlir::Operation * {
      auto *op = v.getDefiningOp();
      while (auto conv = mlir::dyn_cast_or_null<fir::ConvertOp>(op))
        op = conv.getValue().getDefiningOp();
      return op;
    };

    auto isPositiveConstant = [](mlir::Value v) -> bool {
      if (auto conOp =
              mlir::dyn_cast<mlir::arith::ConstantOp>(v.getDefiningOp()))
        if (auto iattr = mlir::dyn_cast<mlir::IntegerAttr>(conOp.getValue()))
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

  des1It = des1.getIndices().begin();
  des2It = des2.getIndices().begin();
  for (bool isTriplet : des1.getIsTriplet()) {
    if (isTriplet) {
      mlir::Value des1Lb = *des1It++;
      mlir::Value des1Ub = *des1It++;
      mlir::Value des2Lb = *des2It++;
      mlir::Value des2Ub = *des2It++;
      // Ignore strides.
      ++des1It;
      ++des2It;
      if (displacedByConstant(des1Ub, des2Lb) ||
          displacedByConstant(des2Ub, des1Lb))
        return true;
    } else {
      ++des1It;
      ++des2It;
    }
  }

  return false;
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
  if (!arrayType)
    return std::nullopt;

  // require that the array elements are trivial
  // TODO: this is just to make the pass easier to think about. Not an inherent
  // limitation
  mlir::Type eleTy = hlfir::getFortranElementType(arrayType);
  if (!fir::isa_trivial(eleTy))
    return std::nullopt;

  // the array must have the same shape as the elemental. CSE should have
  // deduplicated the fir.shape operations where they are provably the same
  // so we just have to check for the same ssa value
  // TODO: add more ways of getting the shape of the array
  mlir::Value arrayShape;
  if (match.array.getDefiningOp())
    arrayShape =
        mlir::TypeSwitch<mlir::Operation *, mlir::Value>(
            match.array.getDefiningOp())
            .Case([](hlfir::DesignateOp designate) {
              return designate.getShape();
            })
            .Case([](hlfir::DeclareOp declare) { return declare.getShape(); })
            .Default([](mlir::Operation *) { return mlir::Value{}; });
  if (!arrayShape) {
    LLVM_DEBUG(llvm::dbgs() << "Can't get shape of " << match.array << " at "
                            << elemental->getLoc() << "\n");
    return std::nullopt;
  }
  if (arrayShape != elemental.getShape()) {
    // f2018 10.2.1.2 (3) requires the lhs and rhs of an assignment to be
    // conformable unless the lhs is an allocatable array. In HLFIR we can
    // see this from the presence or absence of the realloc attribute on
    // hlfir.assign. If it is not a realloc assignment, we can trust that
    // the shapes do conform
    if (match.assign.getRealloc())
      return std::nullopt;
  }

  // the transformation wants to apply the elemental in a do-loop at the
  // hlfir.assign, check there are no effects which make this unsafe

  // keep track of any values written to in the elemental, as these can't be
  // read from between the elemental and the assignment
  // likewise, values read in the elemental cannot be written to between the
  // elemental and the assign
  mlir::SmallVector<mlir::Value, 1> notToBeAccessedBeforeAssign;
  // any accesses to the array between the array and the assignment means it
  // would be unsafe to move the elemental to the assignment
  notToBeAccessedBeforeAssign.push_back(match.array);

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
      if (mlir::isa<mlir::MemoryEffects::Write, mlir::MemoryEffects::Read>(
              effect.getEffect()))
        if (effect.getValue())
          notToBeAccessedBeforeAssign.push_back(effect.getValue());

      // this is safe in the elemental
      continue;
    }

    // don't allow any aliasing writes in the elemental
    if (mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
      LLVM_DEBUG(llvm::dbgs() << "write inside the elemental body\n");
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
        if (!areIdenticalOrDisjointSlices(match.array, designate.getMemref())) {
          LLVM_DEBUG(llvm::dbgs() << "possible read conflict: " << designate
                                  << " at " << elemental.getLoc() << "\n");
          return std::nullopt;
        }
        auto indices = designate.getIndices();
        auto elementalIndices = elemental.getIndices();
        if (indices.size() != elementalIndices.size()) {
          LLVM_DEBUG(llvm::dbgs() << "possible read conflict: " << designate
                                  << " at " << elemental.getLoc() << "\n");
          return std::nullopt;
        }
        if (std::equal(indices.begin(), indices.end(), elementalIndices.begin(),
                       elementalIndices.end()))
          continue;
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
                   << "diasllowed side-effect: " << effect.getValue() << " for "
                   << elemental.getLoc() << "\n");
        return std::nullopt;
      }
    }
  }

  return match;
}

mlir::LogicalResult ElementalAssignBufferization::matchAndRewrite(
    hlfir::ElementalOp elemental, mlir::PatternRewriter &rewriter) const {
  std::optional<MatchInfo> match = findMatch(elemental);
  if (!match)
    return rewriter.notifyMatchFailure(
        elemental, "cannot prove safety of ElementalAssignBufferization");

  mlir::Location loc = elemental->getLoc();
  fir::FirOpBuilder builder(rewriter, elemental.getOperation());
  auto extents = hlfir::getIndexExtents(loc, builder, elemental.getShape());

  // create the loop at the assignment
  builder.setInsertionPoint(match->assign);

  // Generate a loop nest looping around the hlfir.elemental shape and clone
  // hlfir.elemental region inside the inner loop
  hlfir::LoopNest loopNest =
      hlfir::genLoopNest(loc, builder, extents, !elemental.isOrdered());
  builder.setInsertionPointToStart(loopNest.innerLoop.getBody());
  auto yield = hlfir::inlineElementalOp(loc, builder, elemental,
                                        loopNest.oneBasedIndices);
  hlfir::Entity elementValue{yield.getElementValue()};
  rewriter.eraseOp(yield);

  // Assign the element value to the array element for this iteration.
  auto arrayElement = hlfir::getElementAt(
      loc, builder, hlfir::Entity{match->array}, loopNest.oneBasedIndices);
  builder.create<hlfir::AssignOp>(
      loc, elementValue, arrayElement, /*realloc=*/false,
      /*keep_lhs_length_if_realloc=*/false, match->assign.getTemporaryLhs());

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

  mlir::LogicalResult
  matchAndRewrite(hlfir::AssignOp assign,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult BroadcastAssignBufferization::matchAndRewrite(
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
  hlfir::LoopNest loopNest =
      hlfir::genLoopNest(loc, builder, extents, /*isUnordered=*/true);
  builder.setInsertionPointToStart(loopNest.innerLoop.getBody());
  auto arrayElement =
      hlfir::getElementAt(loc, builder, lhs, loopNest.oneBasedIndices);
  builder.create<hlfir::AssignOp>(loc, rhs, arrayElement);
  rewriter.eraseOp(assign);
  return mlir::success();
}

/// Expand hlfir.assign of array RHS to array LHS into a loop nest
/// of element-by-element assignments:
///   hlfir.assign %4 to %5 : !fir.ref<!fir.array<3x3xf32>>,
///                           !fir.ref<!fir.array<3x3xf32>>
/// into:
///   fir.do_loop %arg1 = %c1 to %c3 step %c1 unordered {
///     fir.do_loop %arg2 = %c1 to %c3 step %c1 unordered {
///       %6 = hlfir.designate %4 (%arg2, %arg1)  :
///           (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
///       %7 = fir.load %6 : !fir.ref<f32>
///       %8 = hlfir.designate %5 (%arg2, %arg1)  :
///           (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
///       hlfir.assign %7 to %8 : f32, !fir.ref<f32>
///     }
///   }
///
/// The transformation is correct only when LHS and RHS do not alias.
/// This transformation does not support runtime checking for
/// non-conforming LHS/RHS arrays' shapes currently.
class VariableAssignBufferization
    : public mlir::OpRewritePattern<hlfir::AssignOp> {
private:
public:
  using mlir::OpRewritePattern<hlfir::AssignOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hlfir::AssignOp assign,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult VariableAssignBufferization::matchAndRewrite(
    hlfir::AssignOp assign, mlir::PatternRewriter &rewriter) const {
  if (assign.isAllocatableAssignment())
    return rewriter.notifyMatchFailure(assign, "AssignOp may imply allocation");

  hlfir::Entity rhs{assign.getRhs()};
  // TODO: ExprType check is here to avoid conflicts with
  // ElementalAssignBufferization pattern. We need to combine
  // these matchers into a single one that applies to AssignOp.
  if (mlir::isa<hlfir::ExprType>(rhs.getType()))
    return rewriter.notifyMatchFailure(assign, "RHS is not in memory");

  if (!rhs.isArray())
    return rewriter.notifyMatchFailure(assign,
                                       "AssignOp's RHS is not an array");

  mlir::Type rhsEleTy = rhs.getFortranElementType();
  if (!fir::isa_trivial(rhsEleTy))
    return rewriter.notifyMatchFailure(
        assign, "AssignOp's RHS data type is not trivial");

  hlfir::Entity lhs{assign.getLhs()};
  if (!lhs.isArray())
    return rewriter.notifyMatchFailure(assign,
                                       "AssignOp's LHS is not an array");

  mlir::Type lhsEleTy = lhs.getFortranElementType();
  if (!fir::isa_trivial(lhsEleTy))
    return rewriter.notifyMatchFailure(
        assign, "AssignOp's LHS data type is not trivial");

  if (lhsEleTy != rhsEleTy)
    return rewriter.notifyMatchFailure(assign,
                                       "RHS/LHS element types mismatch");

  fir::AliasAnalysis aliasAnalysis;
  mlir::AliasResult aliasRes = aliasAnalysis.alias(lhs, rhs);
  // TODO: use areIdenticalOrDisjointSlices() to check if
  // we can still do the expansion.
  if (!aliasRes.isNo()) {
    LLVM_DEBUG(llvm::dbgs() << "VariableAssignBufferization:\n"
                            << "\tLHS: " << lhs << "\n"
                            << "\tRHS: " << rhs << "\n"
                            << "\tALIAS: " << aliasRes << "\n");
    return rewriter.notifyMatchFailure(assign, "RHS/LHS may alias");
  }

  mlir::Location loc = assign->getLoc();
  fir::FirOpBuilder builder(rewriter, assign.getOperation());
  builder.setInsertionPoint(assign);
  rhs = hlfir::derefPointersAndAllocatables(loc, builder, rhs);
  lhs = hlfir::derefPointersAndAllocatables(loc, builder, lhs);
  mlir::Value shape = hlfir::genShape(loc, builder, lhs);
  llvm::SmallVector<mlir::Value> extents =
      hlfir::getIndexExtents(loc, builder, shape);
  hlfir::LoopNest loopNest =
      hlfir::genLoopNest(loc, builder, extents, /*isUnordered=*/true);
  builder.setInsertionPointToStart(loopNest.innerLoop.getBody());
  auto rhsArrayElement =
      hlfir::getElementAt(loc, builder, rhs, loopNest.oneBasedIndices);
  rhsArrayElement = hlfir::loadTrivialScalar(loc, builder, rhsArrayElement);
  auto lhsArrayElement =
      hlfir::getElementAt(loc, builder, lhs, loopNest.oneBasedIndices);
  builder.create<hlfir::AssignOp>(loc, rhsArrayElement, lhsArrayElement);
  rewriter.eraseOp(assign);
  return mlir::success();
}

using GenBodyFn =
    std::function<mlir::Value(fir::FirOpBuilder &, mlir::Location, mlir::Value,
                              const llvm::SmallVectorImpl<mlir::Value> &)>;
static mlir::Value generateReductionLoop(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Value init,
                                         mlir::Value shape, GenBodyFn genBody) {
  auto extents = hlfir::getIndexExtents(loc, builder, shape);
  mlir::Value reduction = init;
  mlir::IndexType idxTy = builder.getIndexType();
  mlir::Value oneIdx = builder.createIntegerConstant(loc, idxTy, 1);

  // Create a reduction loop nest. We use one-based indices so that they can be
  // passed to the elemental, and reverse the order so that they can be
  // generated in column-major order for better performance.
  llvm::SmallVector<mlir::Value> indices(extents.size(), mlir::Value{});
  for (unsigned i = 0; i < extents.size(); ++i) {
    auto loop = builder.create<fir::DoLoopOp>(
        loc, oneIdx, extents[extents.size() - i - 1], oneIdx, false,
        /*finalCountValue=*/false, reduction);
    reduction = loop.getRegionIterArgs()[0];
    indices[extents.size() - i - 1] = loop.getInductionVar();
    // Set insertion point to the loop body so that the next loop
    // is inserted inside the current one.
    builder.setInsertionPointToStart(loop.getBody());
  }

  // Generate the body
  reduction = genBody(builder, loc, reduction, indices);

  // Unwind the loop nest.
  for (unsigned i = 0; i < extents.size(); ++i) {
    auto result = builder.create<fir::ResultOp>(loc, reduction);
    auto loop = mlir::cast<fir::DoLoopOp>(result->getParentOp());
    reduction = loop.getResult(0);
    // Set insertion point after the loop operation that we have
    // just processed.
    builder.setInsertionPointAfter(loop.getOperation());
  }

  return reduction;
}

/// Given a reduction operation with an elemental mask, attempt to generate a
/// do-loop to perform the operation inline.
///   %e = hlfir.elemental %shape unordered
///   %r = hlfir.count %e
/// =>
///   %r = for.do_loop %arg = 1 to bound(%shape) step 1 iter_args(%arg2 = init)
///     %i = <inline elemental>
///     %c = <reduce count> %i
///     fir.result %c
template <typename Op>
class ReductionElementalConversion : public mlir::OpRewritePattern<Op> {
public:
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    hlfir::ElementalOp elemental =
        op.getMask().template getDefiningOp<hlfir::ElementalOp>();
    if (!elemental || op.getDim())
      return rewriter.notifyMatchFailure(op, "Did not find valid elemental");

    fir::KindMapping kindMap =
        fir::getKindMapping(op->template getParentOfType<mlir::ModuleOp>());
    fir::FirOpBuilder builder{op, kindMap};

    mlir::Value init;
    GenBodyFn genBodyFn;
    if constexpr (std::is_same_v<Op, hlfir::AnyOp>) {
      init = builder.createIntegerConstant(loc, builder.getI1Type(), 0);
      genBodyFn = [elemental](fir::FirOpBuilder builder, mlir::Location loc,
                              mlir::Value reduction,
                              const llvm::SmallVectorImpl<mlir::Value> &indices)
          -> mlir::Value {
        // Inline the elemental and get the condition from it.
        auto yield = inlineElementalOp(loc, builder, elemental, indices);
        mlir::Value cond = builder.create<fir::ConvertOp>(
            loc, builder.getI1Type(), yield.getElementValue());
        yield->erase();

        // Conditionally set the reduction variable.
        return builder.create<mlir::arith::OrIOp>(loc, reduction, cond);
      };
    } else if constexpr (std::is_same_v<Op, hlfir::AllOp>) {
      init = builder.createIntegerConstant(loc, builder.getI1Type(), 1);
      genBodyFn = [elemental](fir::FirOpBuilder builder, mlir::Location loc,
                              mlir::Value reduction,
                              const llvm::SmallVectorImpl<mlir::Value> &indices)
          -> mlir::Value {
        // Inline the elemental and get the condition from it.
        auto yield = inlineElementalOp(loc, builder, elemental, indices);
        mlir::Value cond = builder.create<fir::ConvertOp>(
            loc, builder.getI1Type(), yield.getElementValue());
        yield->erase();

        // Conditionally set the reduction variable.
        return builder.create<mlir::arith::AndIOp>(loc, reduction, cond);
      };
    } else if constexpr (std::is_same_v<Op, hlfir::CountOp>) {
      init = builder.createIntegerConstant(loc, op.getType(), 0);
      genBodyFn = [elemental](fir::FirOpBuilder builder, mlir::Location loc,
                              mlir::Value reduction,
                              const llvm::SmallVectorImpl<mlir::Value> &indices)
          -> mlir::Value {
        // Inline the elemental and get the condition from it.
        auto yield = inlineElementalOp(loc, builder, elemental, indices);
        mlir::Value cond = builder.create<fir::ConvertOp>(
            loc, builder.getI1Type(), yield.getElementValue());
        yield->erase();

        // Conditionally add one to the current value
        mlir::Value one =
            builder.createIntegerConstant(loc, reduction.getType(), 1);
        mlir::Value add1 =
            builder.create<mlir::arith::AddIOp>(loc, reduction, one);
        return builder.create<mlir::arith::SelectOp>(loc, cond, add1,
                                                     reduction);
      };
    } else {
      return mlir::failure();
    }

    mlir::Value res = generateReductionLoop(builder, loc, init,
                                            elemental.getOperand(0), genBodyFn);
    if (res.getType() != op.getType())
      res = builder.create<fir::ConvertOp>(loc, op.getType(), res);

    // Check if the op was the only user of the elemental (apart from a
    // destroy), and remove it if so.
    mlir::Operation::user_range elemUsers = elemental->getUsers();
    hlfir::DestroyOp elemDestroy;
    if (std::distance(elemUsers.begin(), elemUsers.end()) == 2) {
      elemDestroy = mlir::dyn_cast<hlfir::DestroyOp>(*elemUsers.begin());
      if (!elemDestroy)
        elemDestroy = mlir::dyn_cast<hlfir::DestroyOp>(*++elemUsers.begin());
    }

    rewriter.replaceOp(op, res);
    if (elemDestroy) {
      rewriter.eraseOp(elemDestroy);
      rewriter.eraseOp(elemental);
    }
    return mlir::success();
  }
};

// Look for minloc(mask=elemental) and generate the minloc loop with
// inlined elemental.
//  %e = hlfir.elemental %shape ({ ... })
//  %m = hlfir.minloc %array mask %e
template <typename Op>
class MinMaxlocElementalConversion : public mlir::OpRewritePattern<Op> {
public:
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op mloc, mlir::PatternRewriter &rewriter) const override {
    if (!mloc.getMask() || mloc.getDim() || mloc.getBack())
      return rewriter.notifyMatchFailure(mloc,
                                         "Did not find valid minloc/maxloc");

    bool isMax = std::is_same_v<Op, hlfir::MaxlocOp>;

    auto elemental =
        mloc.getMask().template getDefiningOp<hlfir::ElementalOp>();
    if (!elemental || hlfir::elementalOpMustProduceTemp(elemental))
      return rewriter.notifyMatchFailure(mloc, "Did not find elemental");

    mlir::Value array = mloc.getArray();

    unsigned rank = mlir::cast<hlfir::ExprType>(mloc.getType()).getShape()[0];
    mlir::Type arrayType = array.getType();
    if (!mlir::isa<fir::BoxType>(arrayType))
      return rewriter.notifyMatchFailure(
          mloc, "Currently requires a boxed type input");
    mlir::Type elementType = hlfir::getFortranElementType(arrayType);
    if (!fir::isa_trivial(elementType))
      return rewriter.notifyMatchFailure(
          mloc, "Character arrays are currently not handled");

    mlir::Location loc = mloc.getLoc();
    fir::FirOpBuilder builder{rewriter, mloc.getOperation()};
    mlir::Value resultArr = builder.createTemporary(
        loc, fir::SequenceType::get(
                 rank, hlfir::getFortranElementType(mloc.getType())));

    auto init = [isMax](fir::FirOpBuilder builder, mlir::Location loc,
                        mlir::Type elementType) {
      if (auto ty = mlir::dyn_cast<mlir::FloatType>(elementType)) {
        const llvm::fltSemantics &sem = ty.getFloatSemantics();
        llvm::APFloat limit = llvm::APFloat::getInf(sem, /*Negative=*/isMax);
        return builder.createRealConstant(loc, elementType, limit);
      }
      unsigned bits = elementType.getIntOrFloatBitWidth();
      int64_t limitInt =
          isMax ? llvm::APInt::getSignedMinValue(bits).getSExtValue()
                : llvm::APInt::getSignedMaxValue(bits).getSExtValue();
      return builder.createIntegerConstant(loc, elementType, limitInt);
    };

    auto genBodyOp =
        [&rank, &resultArr, &elemental, isMax](
            fir::FirOpBuilder builder, mlir::Location loc,
            mlir::Type elementType, mlir::Value array, mlir::Value flagRef,
            mlir::Value reduction,
            const llvm::SmallVectorImpl<mlir::Value> &indices) -> mlir::Value {
      // We are in the innermost loop: generate the elemental inline
      mlir::Value oneIdx =
          builder.createIntegerConstant(loc, builder.getIndexType(), 1);
      llvm::SmallVector<mlir::Value> oneBasedIndices;
      llvm::transform(
          indices, std::back_inserter(oneBasedIndices), [&](mlir::Value V) {
            return builder.create<mlir::arith::AddIOp>(loc, V, oneIdx);
          });
      hlfir::YieldElementOp yield =
          hlfir::inlineElementalOp(loc, builder, elemental, oneBasedIndices);
      mlir::Value maskElem = yield.getElementValue();
      yield->erase();

      mlir::Type ifCompatType = builder.getI1Type();
      mlir::Value ifCompatElem =
          builder.create<fir::ConvertOp>(loc, ifCompatType, maskElem);

      llvm::SmallVector<mlir::Type> resultsTy = {elementType, elementType};
      fir::IfOp maskIfOp =
          builder.create<fir::IfOp>(loc, elementType, ifCompatElem,
                                    /*withElseRegion=*/true);
      builder.setInsertionPointToStart(&maskIfOp.getThenRegion().front());

      // Set flag that mask was true at some point
      mlir::Value flagSet = builder.createIntegerConstant(
          loc, mlir::cast<fir::ReferenceType>(flagRef.getType()).getEleTy(), 1);
      mlir::Value isFirst = builder.create<fir::LoadOp>(loc, flagRef);
      mlir::Value addr = hlfir::getElementAt(loc, builder, hlfir::Entity{array},
                                             oneBasedIndices);
      mlir::Value elem = builder.create<fir::LoadOp>(loc, addr);

      // Compare with the max reduction value
      mlir::Value cmp;
      if (mlir::isa<mlir::FloatType>(elementType)) {
        // For FP reductions we want the first smallest value to be used, that
        // is not NaN. A OGL/OLT condition will usually work for this unless all
        // the values are Nan or Inf. This follows the same logic as
        // NumericCompare for Minloc/Maxlox in extrema.cpp.
        cmp = builder.create<mlir::arith::CmpFOp>(
            loc,
            isMax ? mlir::arith::CmpFPredicate::OGT
                  : mlir::arith::CmpFPredicate::OLT,
            elem, reduction);

        mlir::Value cmpNan = builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::UNE, reduction, reduction);
        mlir::Value cmpNan2 = builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OEQ, elem, elem);
        cmpNan = builder.create<mlir::arith::AndIOp>(loc, cmpNan, cmpNan2);
        cmp = builder.create<mlir::arith::OrIOp>(loc, cmp, cmpNan);
      } else if (mlir::isa<mlir::IntegerType>(elementType)) {
        cmp = builder.create<mlir::arith::CmpIOp>(
            loc,
            isMax ? mlir::arith::CmpIPredicate::sgt
                  : mlir::arith::CmpIPredicate::slt,
            elem, reduction);
      } else {
        llvm_unreachable("unsupported type");
      }

      // The condition used for the loop is isFirst || <the condition above>.
      isFirst = builder.create<fir::ConvertOp>(loc, cmp.getType(), isFirst);
      isFirst = builder.create<mlir::arith::XOrIOp>(
          loc, isFirst, builder.createIntegerConstant(loc, cmp.getType(), 1));
      cmp = builder.create<mlir::arith::OrIOp>(loc, cmp, isFirst);

      // Set the new coordinate to the result
      fir::IfOp ifOp = builder.create<fir::IfOp>(loc, elementType, cmp,
                                                 /*withElseRegion*/ true);

      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      builder.create<fir::StoreOp>(loc, flagSet, flagRef);
      mlir::Type resultElemTy =
          hlfir::getFortranElementType(resultArr.getType());
      mlir::Type returnRefTy = builder.getRefType(resultElemTy);
      mlir::IndexType idxTy = builder.getIndexType();

      for (unsigned int i = 0; i < rank; ++i) {
        mlir::Value index = builder.createIntegerConstant(loc, idxTy, i + 1);
        mlir::Value resultElemAddr = builder.create<hlfir::DesignateOp>(
            loc, returnRefTy, resultArr, index);
        mlir::Value fortranIndex = builder.create<fir::ConvertOp>(
            loc, resultElemTy, oneBasedIndices[i]);
        builder.create<fir::StoreOp>(loc, fortranIndex, resultElemAddr);
      }
      builder.create<fir::ResultOp>(loc, elem);
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      builder.create<fir::ResultOp>(loc, reduction);
      builder.setInsertionPointAfter(ifOp);

      // Close the mask if
      builder.create<fir::ResultOp>(loc, ifOp.getResult(0));
      builder.setInsertionPointToStart(&maskIfOp.getElseRegion().front());
      builder.create<fir::ResultOp>(loc, reduction);
      builder.setInsertionPointAfter(maskIfOp);

      return maskIfOp.getResult(0);
    };
    auto getAddrFn = [](fir::FirOpBuilder builder, mlir::Location loc,
                        const mlir::Type &resultElemType, mlir::Value resultArr,
                        mlir::Value index) {
      mlir::Type resultRefTy = builder.getRefType(resultElemType);
      mlir::Value oneIdx =
          builder.createIntegerConstant(loc, builder.getIndexType(), 1);
      index = builder.create<mlir::arith::AddIOp>(loc, index, oneIdx);
      return builder.create<hlfir::DesignateOp>(loc, resultRefTy, resultArr,
                                                index);
    };

    // Initialize the result
    mlir::Type resultElemTy = hlfir::getFortranElementType(resultArr.getType());
    mlir::Type resultRefTy = builder.getRefType(resultElemTy);
    mlir::Value returnValue =
        builder.createIntegerConstant(loc, resultElemTy, 0);
    for (unsigned int i = 0; i < rank; ++i) {
      mlir::Value index =
          builder.createIntegerConstant(loc, builder.getIndexType(), i + 1);
      mlir::Value resultElemAddr = builder.create<hlfir::DesignateOp>(
          loc, resultRefTy, resultArr, index);
      builder.create<fir::StoreOp>(loc, returnValue, resultElemAddr);
    }

    fir::genMinMaxlocReductionLoop(builder, array, init, genBodyOp, getAddrFn,
                                   rank, elementType, loc, builder.getI1Type(),
                                   resultArr, false);

    mlir::Value asExpr = builder.create<hlfir::AsExprOp>(
        loc, resultArr, builder.createBool(loc, false));

    // Check all the users - the destroy is no longer required, and any assign
    // can use resultArr directly so that VariableAssignBufferization in this
    // pass can optimize the results. Other operations are replaces with an
    // AsExpr for the temporary resultArr.
    llvm::SmallVector<hlfir::DestroyOp> destroys;
    llvm::SmallVector<hlfir::AssignOp> assigns;
    for (auto user : mloc->getUsers()) {
      if (auto destroy = mlir::dyn_cast<hlfir::DestroyOp>(user))
        destroys.push_back(destroy);
      else if (auto assign = mlir::dyn_cast<hlfir::AssignOp>(user))
        assigns.push_back(assign);
    }

    // Check if the minloc/maxloc was the only user of the elemental (apart from
    // a destroy), and remove it if so.
    mlir::Operation::user_range elemUsers = elemental->getUsers();
    hlfir::DestroyOp elemDestroy;
    if (std::distance(elemUsers.begin(), elemUsers.end()) == 2) {
      elemDestroy = mlir::dyn_cast<hlfir::DestroyOp>(*elemUsers.begin());
      if (!elemDestroy)
        elemDestroy = mlir::dyn_cast<hlfir::DestroyOp>(*++elemUsers.begin());
    }

    for (auto d : destroys)
      rewriter.eraseOp(d);
    for (auto a : assigns)
      a.setOperand(0, resultArr);
    rewriter.replaceOp(mloc, asExpr);
    if (elemDestroy) {
      rewriter.eraseOp(elemDestroy);
      rewriter.eraseOp(elemental);
    }
    return mlir::success();
  }
};

class OptimizedBufferizationPass
    : public hlfir::impl::OptimizedBufferizationBase<
          OptimizedBufferizationPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();

    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks
    config.enableRegionSimplification = false;

    mlir::RewritePatternSet patterns(context);
    // TODO: right now the patterns are non-conflicting,
    // but it might be better to run this pass on hlfir.assign
    // operations and decide which transformation to apply
    // at one place (e.g. we may use some heuristics and
    // choose different optimization strategies).
    // This requires small code reordering in ElementalAssignBufferization.
    patterns.insert<ElementalAssignBufferization>(context);
    patterns.insert<BroadcastAssignBufferization>(context);
    patterns.insert<VariableAssignBufferization>(context);
    patterns.insert<ReductionElementalConversion<hlfir::CountOp>>(context);
    patterns.insert<ReductionElementalConversion<hlfir::AnyOp>>(context);
    patterns.insert<ReductionElementalConversion<hlfir::AllOp>>(context);
    patterns.insert<MinMaxlocElementalConversion<hlfir::MinlocOp>>(context);
    patterns.insert<MinMaxlocElementalConversion<hlfir::MaxlocOp>>(context);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns), config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in HLFIR optimized bufferization");
      signalPassFailure();
    }
  }
};
} // namespace
