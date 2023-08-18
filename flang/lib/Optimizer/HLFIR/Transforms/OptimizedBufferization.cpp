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

std::optional<ElementalAssignBufferization::MatchInfo>
ElementalAssignBufferization::findMatch(hlfir::ElementalOp elemental) {
  mlir::Operation::user_range users = elemental->getUsers();
  // the only uses of the elemental should be the assignment and the destroy
  if (std::distance(users.begin(), users.end()) != 2) {
    LLVM_DEBUG(llvm::dbgs() << "Too many uses of the elemental\n");
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
        if (designate.getMemref() != match.array) {
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
    LLVM_DEBUG(llvm::dbgs() << "diasllowed side-effect: " << effect.getValue()
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

class OptimizedBufferizationPass
    : public hlfir::impl::OptimizedBufferizationBase<
          OptimizedBufferizationPass> {
public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext *context = &getContext();

    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks
    config.enableRegionSimplification = false;

    mlir::RewritePatternSet patterns(context);
    patterns.insert<ElementalAssignBufferization>(context);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            func, std::move(patterns), config))) {
      mlir::emitError(func.getLoc(),
                      "failure in HLFIR optimized bufferization");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> hlfir::createOptimizedBufferizationPass() {
  return std::make_unique<OptimizedBufferizationPass>();
}
