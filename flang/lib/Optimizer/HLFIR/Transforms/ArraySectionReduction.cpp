//===- ArraySectionReduction.cpp - Promote array-section reductions -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass implementing the "loop-invariant array-section reduction promotion"
// transform.
//
// It looks for the pattern:
//
//     a(:,i) = 0.0                        ! (init) whole-section define
//     do j = ...
//        if (...) a(:,i) = a(:,i) + ...   ! (RMW)  invariant-section reduction
//     end do
//     ... = f(a(:,i))                     ! (use)  post-loop consume
//
// When preconditions hold, it promotes the section to a constant-shape local
// temporary T: fold the init into T, redirect the in-loop read-modify-write to
// T, and copy T back to the section after the loop (see promote()). This turns
// the in-loop store to a loop-invariant descriptor section into a reduction
// into a local fixed-size array, removing the memory dependence that blocks
// LLVM's loop vectorizer.
//
// Preconditions checked:
//   P0  the section element type is a trivial value type (numeric or logical)
//       and non-volatile; character, derived, polymorphic, and volatile
//       sections are not handled,
//   P1  the section address is loop-invariant in the enclosing loop (no store
//       or call inside the loop rewrites the memory its subscripts read),
//   P2  an unconditional store to the identical whole section dominates the
//       loop (proven by structural section-equivalence + dominance),
//   P3  every other access to the section inside the loop is redirected to the
//       temporary: no write may alias it, no read may bypass it, and no call
//       may modify or read its storage (fir::AliasAnalysis),
//   P4  the RHS carries a compile-time-constant shape whose extent is small
//       enough to scalarize into register accumulators; larger or
//       runtime-shaped sections are detected but not rewritten.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

namespace hlfir {
#define GEN_PASS_DEF_ARRAYSECTIONREDUCTION
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

#define DEBUG_TYPE "array-section-reduction"

namespace {

/// Strip no-op address/value forwarding to get at the underlying SSA value.
static mlir::Value stripConvert(mlir::Value v) {
  while (v) {
    if (auto conv = v.getDefiningOp<fir::ConvertOp>()) {
      v = conv.getValue();
      continue;
    }
    break;
  }
  return v;
}

/// Strip hlfir.declare / fir.declare so two references to the same declared
/// entity compare equal even when reached through different declare results.
static mlir::Value stripDeclare(mlir::Value v) {
  v = stripConvert(v);
  if (auto res = mlir::dyn_cast<mlir::OpResult>(v)) {
    mlir::Operation *def = res.getOwner();
    if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(def))
      return d.getMemref();
    if (auto d = mlir::dyn_cast<fir::DeclareOp>(def))
      return d.getMemref();
  }
  return v;
}

/// Structural value-equivalence: true when \p a and \p b provably evaluate to
/// the same runtime value. Handles the address/shape computations the front end
/// emits for section designators (loads, box_dims, designates, shapes, arith).
/// \p depth caps the recursion into operand trees; exceeding it fails closed
/// (no match -> no promotion), which is always sound.
static constexpr unsigned maxEquivRecursionDepth = 32;
static bool equiv(mlir::Value a, mlir::Value b, unsigned depth = 0) {
  if (depth > maxEquivRecursionDepth) // guard pathologically nested designators
    return false;
  a = stripConvert(a);
  b = stripConvert(b);
  if (!a || !b)
    return false;
  if (a == b)
    return true;

  // Two loads match by their declared address, ignoring any store between them.
  // Sound only because the caller proves the loaded value is unchanged between
  // the two program points: P1 (no in-loop write to the address deps) and the
  // P2 interval check (section and subscripts untouched from init to loop).
  auto la = a.getDefiningOp<fir::LoadOp>();
  auto lb = b.getDefiningOp<fir::LoadOp>();
  if (la || lb)
    return la && lb &&
           stripDeclare(la.getMemref()) == stripDeclare(lb.getMemref());

  auto ra = mlir::dyn_cast<mlir::OpResult>(a);
  auto rb = mlir::dyn_cast<mlir::OpResult>(b);
  if (!ra || !rb || ra.getResultNumber() != rb.getResultNumber())
    return false;

  // Only match address/shape/index computations. hlfir.designate and
  // fir.box_dims are pure address/descriptor-inspection ops, so allow them.
  auto isAddressOrShapeOp = [](mlir::Operation *op) {
    return mlir::isMemoryEffectFree(op) ||
           mlir::isa<fir::BoxDimsOp, fir::ShapeOp, fir::ShapeShiftOp,
                     hlfir::DesignateOp>(op);
  };
  if (!isAddressOrShapeOp(ra.getOwner()) || !isAddressOrShapeOp(rb.getOwner()))
    return false;

  // Structural match only: commutativity is not modeled (i+1 vs 1+i fails to
  // match), which costs a missed promotion but never a wrong one.
  return mlir::OperationEquivalence::isEquivalentTo(
      ra.getOwner(), rb.getOwner(),
      /*checkEquivalent=*/
      [depth](mlir::Value oa, mlir::Value ob) {
        return mlir::success(equiv(oa, ob, depth + 1));
      },
      /*markEquivalent=*/nullptr,
      mlir::OperationEquivalence::Flags::IgnoreLocations);
}

/// True when two hlfir.designate results address the identical array section.
static bool sameSection(hlfir::DesignateOp d1, hlfir::DesignateOp d2) {
  return equiv(d1.getResult(), d2.getResult());
}

/// Collect the hlfir.designate reads of the same section as \p lhsDes anywhere
/// in the expression producing \p rhs (including nested hlfir.elemental ops).
/// Matched by section identity, not SSA id. True if any read was found (RMW).
static bool
collectSectionReads(mlir::Value rhs, hlfir::DesignateOp lhsDes,
                    llvm::SmallVectorImpl<hlfir::DesignateOp> &reads) {
  mlir::Operation *rhsDef = rhs.getDefiningOp();
  if (!rhsDef)
    return false;
  llvm::SmallPtrSet<mlir::Operation *, 4> seen;
  rhsDef->walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands())
      if (auto d = operand.getDefiningOp<hlfir::DesignateOp>())
        if (sameSection(d, lhsDes) && seen.insert(d).second)
          reads.push_back(d);
  });
  return !reads.empty();
}

/// Collect the declared addresses a section designator loads to form its
/// address (e.g. the index i in a(:,i)), by walking its operand cone for loads.
static llvm::SmallVector<mlir::Value>
collectAddressDeps(hlfir::DesignateOp lhsDes) {
  llvm::SmallVector<mlir::Value> addrDeps;
  llvm::SmallPtrSet<mlir::Value, 8> seenDeps;
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;
  llvm::SmallVector<mlir::Value, 16> worklist;
  for (mlir::Value operand : lhsDes.getOperation()->getOperands())
    worklist.push_back(operand);
  while (!worklist.empty()) {
    mlir::Operation *def = worklist.pop_back_val().getDefiningOp();
    if (!def || !visited.insert(def).second)
      continue;
    if (auto load = mlir::dyn_cast<fir::LoadOp>(def)) {
      // Keep the [hl]fir.declare address: AliasAnalysis classifies the variable
      // (e.g. an OpenMP-private index) by walking through its declare, so
      // stripping it here would defeat that and force a conservative MayAlias.
      mlir::Value m = load.getMemref();
      if (seenDeps.insert(m).second)
        addrDeps.push_back(m);
    }
    for (mlir::Value operand : def->getOperands())
      worklist.push_back(operand);
  }
  return addrDeps;
}

/// True when \p loop has constant bounds and a trip count at most \p
/// maxTripCount -- the threshold below which we assume the unroller fully
/// unrolls it before the vectorizer runs, so a nested loop no longer blocks
/// annotating the enclosing reduction. This only gates the separable
/// vectorization hint: an over-estimate at most annotates a loop the vectorizer
/// then ignores (a surviving inner loop keeps the outer non-innermost), and an
/// under-estimate omits a hint it could have used -- neither is a correctness
/// change nor a regression.
/// TODO: a target-aware threshold would come from TargetTransformInfo.
static bool isFullyUnrollableLoop(fir::DoLoopOp loop, int64_t maxTripCount) {
  std::optional<int64_t> lb = mlir::getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> ub = mlir::getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
  if (!lb || !ub || !step || *step <= 0)
    return false;
  // Overflow here means an astronomically large loop, i.e. not unrollable.
  int64_t range, numerator;
  if (llvm::SubOverflow(*ub, *lb, range) ||
      llvm::AddOverflow(range, *step, numerator))
    return false;
  int64_t trip = numerator > 0 ? numerator / *step : 0;
  return trip <= maxTripCount;
}

class ArraySectionReductionPass
    : public hlfir::impl::ArraySectionReductionBase<ArraySectionReductionPass> {
public:
  using ArraySectionReductionBase<
      ArraySectionReductionPass>::ArraySectionReductionBase;

  void runOnOperation() override {
    mlir::Operation *root = getOperation();

    // TODO: only fir.do_loop is handled. Extend to other loop carriers
    // (fir.iterate_while, and the OpenMP loop nest) to cover more kernels.
    llvm::SmallVector<fir::DoLoopOp> loops;
    root->walk([&](fir::DoLoopOp loop) { loops.push_back(loop); });
    if (loops.empty())
      return; // no loop to promote into; skip the dominance computation

    // Op-agnostic, like the sibling HLFIR passes: the pipeline runs it per
    // top-level op, so root is usually a func; under fir-opt it may be the
    // module. That stays correct because every dominance query is intra-region
    // (findDominatingFullDef stops at root, never crossing functions) and
    // DominanceInfo is computed lazily per region.
    mlir::DominanceInfo domInfo(root);
    fir::AliasAnalysis aliasAnalysis;
    // domInfo stays valid across the promotions below: promote() only inserts
    // ops and rewrites operands, never changing the CFG (no block add/remove).
    for (fir::DoLoopOp loop : loops)
      matchLoop(loop, domInfo, aliasAnalysis);
  }

private:
  /// Promote every loop-invariant array-section reduction found in \p loop.
  void matchLoop(fir::DoLoopOp loop, mlir::DominanceInfo &domInfo,
                 fir::AliasAnalysis &aliasAnalysis) {
    struct Candidate {
      hlfir::AssignOp init;
      hlfir::AssignOp rmw;
      llvm::SmallVector<hlfir::DesignateOp> reads;
    };
    llvm::SmallVector<Candidate> candidates;

    loop.walk([&](hlfir::AssignOp rmw) {
      hlfir::Entity lhs(rmw.getLhs());
      if (!lhs.isArray())
        return;
      auto lhsDes = lhs.getDefiningOp<hlfir::DesignateOp>();
      if (!lhsDes)
        return;
      LLVM_DEBUG(llvm::dbgs()
                 << "array-section-reduction: candidate array-section assign "
                 << rmw.getLoc() << "\n");

      // RMW reduction: the RHS must read the same section it writes.
      llvm::SmallVector<hlfir::DesignateOp> reads;
      if (!collectSectionReads(rmw.getRhs(), lhsDes, reads)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  reject: RHS does not read the section (not an RMW)\n");
        return;
      }

      // P0: the section element type must be a non-volatile trivial value
      // type.
      mlir::Type eleTy = lhs.getFortranElementType();
      if (!fir::isa_trivial(eleTy)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  reject: non-trivial section element type (P0)\n");
        return;
      }
      if (fir::isa_volatile_type(lhs.getType())) {
        LLVM_DEBUG(llvm::dbgs() << "  reject: volatile section (P0)\n");
        return;
      }

      // P4: the RHS must have a compile-time-constant shape whose extent is
      // small enough to scalarize into register accumulators.
      // maxReductionExtent (the max-reduction-extent option) tracks SIMD
      // register pressure.
      // TODO: derive the bound from TargetTransformInfo when it is available
      // to this pass instead of a fixed default.
      std::optional<int64_t> extent = rhsConstantExtent(rmw.getRhs());
      if (!extent || *extent > maxReductionExtent) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  reject: runtime or too-large section shape (P4)\n");
        return;
      }

      // P2: an unconditional store to the identical whole section dominating L.
      hlfir::AssignOp init = findDominatingFullDef(loop, lhsDes, domInfo);
      if (!init) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  reject: no dominating full-section init (P2)\n");
        return;
      }

      // Address dependences of the section (subscripts it loads, e.g. i).
      llvm::SmallVector<mlir::Value> addrDeps = collectAddressDeps(lhsDes);

      // P1: the section address must be loop-invariant.
      if (!sectionAddressLoopInvariant(loop, addrDeps, aliasAnalysis)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  reject: section address not loop-invariant (P1)\n");
        return;
      }

      // P2 (interval): nothing may access the section between the init and the
      // loop.
      if (!sectionUntouchedBetweenInitAndLoop(init, loop, lhs, addrDeps,
                                              domInfo, aliasAnalysis)) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "  reject: section accessed between init and loop (P2)\n");
        return;
      }

      // P3 (writes): no other write, and no call touching the section, inside
      // L.
      if (!noAliasingWritesInLoop(loop, rmw, lhs, aliasAnalysis)) {
        LLVM_DEBUG(llvm::dbgs() << "  reject: possible aliasing write or call "
                                   "inside the loop (P3)\n");
        return;
      }

      // P3 (reads): every in-loop access to the section must be redirected to
      // T.
      if (!allAliasingAccessesRedirected(loop, lhs, lhsDes, reads,
                                         aliasAnalysis)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  reject: section accessed in the loop outside the "
                      "read-modify-write (P3)\n");
        return;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "  loop-invariant array-section reduction; init dominates "
                    "at "
                 << init.getLoc() << "\n");
      candidates.push_back({init, rmw, std::move(reads)});
    });

    for (Candidate &c : candidates)
      promote(loop, c.init, c.rmw, c.reads);
  }

  /// Rewrite the matched reduction to accumulate into a constant-shape local
  /// temporary T (steps below), removing the in-loop store to the
  /// loop-invariant descriptor section that blocks LLVM's loop vectorizer.
  void promote(fir::DoLoopOp loop, hlfir::AssignOp init, hlfir::AssignOp rmw,
               llvm::ArrayRef<hlfir::DesignateOp> reads) {
    mlir::Location loc = rmw.getLoc();
    // RHS is a constant-shape array expr (P4); use its shape for T.
    auto rhsSeqTy = mlir::cast<fir::SequenceType>(
        hlfir::getFortranElementOrSequenceType(rmw.getRhs().getType()));
    mlir::Type seqTy = fir::SequenceType::get(
        rhsSeqTy.getShape(),
        hlfir::Entity(rmw.getLhs()).getFortranElementType());

    // 1. Allocate T. createTemporary hoists the constant-shape alloca to the
    //    enclosing alloca block, so it is allocated once and stays
    //    thread-local inside an outlined OpenMP region.
    mlir::OpBuilder opBuilder(init);
    fir::FirOpBuilder builder(opBuilder, init.getOperation());
    mlir::Value temp = builder.createTemporary(loc, seqTy, ".array_reduction");

    // Capture the section (e.g. a(:,i)) before step 2 retargets the init;
    // reused as the copy-out target in step 4.
    mlir::Value section = init.getLhs();

    // 2. Fold the init into T.
    init.getLhsMutable().assign(temp);

    // 3. Redirect the in-loop RMW to read and write T.
    for (hlfir::DesignateOp read : reads)
      read.getResult().replaceUsesWithIf(temp, [&](mlir::OpOperand &use) {
        return loop->isProperAncestor(use.getOwner());
      });
    rmw.getLhsMutable().assign(temp);

    // 4. Copy T back to the section after the loop (a(:,i) = T).
    builder.setInsertionPointAfter(loop);
    hlfir::AssignOp::create(builder, loc, temp, section, /*realloc=*/false,
                            /*keep_lhs_length_if_realloc=*/false,
                            /*temporary_lhs=*/false);

    // 5. Force-enable vectorization: once T SROAs into scalar accumulators the
    //    loop is a plain reduction, so overriding the cost model is intended.
    //    Only requested at O2/O3; opt out with -force-vectorize=false.
    if (!forceVectorize)
      return;
    // Hint the innermost loop around the RMW: for a nested reduction the match
    // is at an outer loop but only the inner loop vectorizes.
    fir::DoLoopOp hintLoop = rmw->getParentOfType<fir::DoLoopOp>();
    // Bail if the body carries a loop that keeps the vectorizer out: one that
    // reads the temporary (an inlined sum(a(:,i)) recurrence) or a non-
    // unrollable inner loop. A small constant-trip scratch loop is fine.
    bool blocked = false;
    hintLoop->getRegion(0).walk([&](fir::DoLoopOp nested) {
      bool usesTemp = llvm::any_of(temp.getUsers(), [&](mlir::Operation *user) {
        return nested->isAncestor(user);
      });
      if (usesTemp || !isFullyUnrollableLoop(nested, maxUnrollableTripCount)) {
        blocked = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (blocked)
      return;
    mlir::MLIRContext *ctx = hintLoop.getContext();
    mlir::LLVM::LoopAnnotationAttr existing = hintLoop.getLoopAnnotationAttr();
    mlir::LLVM::LoopVectorizeAttr existingVec =
        existing ? existing.getVectorize() : nullptr;

    // Respect an explicit vectorize `disable = true` (e.g. from !DIR$
    // NOVECTOR).
    if (existingVec && existingVec.getDisable() &&
        existingVec.getDisable().getValue())
      return;

    // Set only the vectorize-enable field, preserving any existing options.
    mlir::BoolAttr enable = mlir::BoolAttr::get(ctx, /*disable=*/false);
    mlir::LLVM::LoopVectorizeAttr vectorize =
        existingVec
            ? mlir::LLVM::LoopVectorizeAttr::get(
                  ctx, enable, existingVec.getPredicateEnable(),
                  existingVec.getScalableEnable(), existingVec.getWidth(),
                  existingVec.getFollowupVectorized(),
                  existingVec.getFollowupEpilogue(),
                  existingVec.getFollowupAll())
            : mlir::LLVM::LoopVectorizeAttr::get(ctx, enable, {}, {}, {}, {},
                                                 {}, {});

    // Merge into any existing annotation, preserving all non-vectorize fields
    mlir::LLVM::LoopAnnotationAttr annotation =
        existing ? mlir::LLVM::LoopAnnotationAttr::get(
                       ctx, existing.getDisableNonforced(), vectorize,
                       existing.getInterleave(), existing.getUnroll(),
                       existing.getUnrollAndJam(), existing.getLicm(),
                       existing.getDistribute(), existing.getPipeline(),
                       existing.getPeeled(), existing.getUnswitch(),
                       existing.getMustProgress(), existing.getIsVectorized(),
                       existing.getStartLoc(), existing.getEndLoc(),
                       existing.getParallelAccesses())
                 : mlir::LLVM::LoopAnnotationAttr::get(ctx, {}, vectorize, {},
                                                       {}, {}, {}, {}, {}, {},
                                                       {}, {}, {}, {}, {}, {});
    hintLoop.setLoopAnnotationAttr(annotation);
  }

  /// Find the nearest unconditional hlfir.assign whose LHS designates the
  /// identical whole section as \p rmwLhs and that dominates \p loop (P2). The
  /// nearest (last dominating) define is required: with repeated same-section
  /// reductions an earlier define could be a prior copy-out, not this loop's
  /// init.
  hlfir::AssignOp findDominatingFullDef(fir::DoLoopOp loop,
                                        hlfir::DesignateOp rmwLhs,
                                        mlir::DominanceInfo &domInfo) {
    auto scanBlock = [&](mlir::Block *b) -> hlfir::AssignOp {
      hlfir::AssignOp found;
      for (mlir::Operation &op : *b) {
        auto cand = mlir::dyn_cast<hlfir::AssignOp>(op);
        if (!cand)
          continue;
        auto candLhs = cand.getLhs().getDefiningOp<hlfir::DesignateOp>();
        if (!candLhs || !domInfo.dominates(cand, loop))
          continue;
        if (sameSection(candLhs, rmwLhs))
          found = cand; // keep the last (nearest to the loop) match
      }
      return found;
    };

    // Only ops that dominate the loop can be the init, so restrict the search
    // to the loop's block and its dominator-tree ancestors, walking out through
    // the enclosing regions, instead of re-walking the whole function per
    // candidate.
    mlir::Operation *root = getOperation();
    for (mlir::Operation *entry = loop.getOperation(); entry;) {
      mlir::Block *block = entry->getBlock();
      if (!block)
        break;
      mlir::Region *region = block->getParent();
      // getNode asserts on single-block regions (its dom tree is not built);
      // there the block is its own only dominator, so scan it directly.
      if (region && !region->hasOneBlock()) {
        for (mlir::DominanceInfoNode *node = domInfo.getNode(block); node;
             node = node->getIDom())
          if (hlfir::AssignOp init = scanBlock(node->getBlock()))
            return init;
      } else if (hlfir::AssignOp init = scanBlock(block)) {
        return init;
      }
      if (entry == root || !region)
        break;
      entry = region->getParentOp();
    }
    return {};
  }

  /// P1: return true when the section's address is loop-invariant in \p loop,
  /// i.e. no in-loop store or call may write a location its subscripts load
  /// (e.g. the index i in a(:,i)). Not covered by P2, which matches addresses
  /// structurally but not whether they are rewritten mid-loop.
  bool sectionAddressLoopInvariant(fir::DoLoopOp loop,
                                   llvm::ArrayRef<mlir::Value> addrDeps,
                                   fir::AliasAnalysis &aliasAnalysis) {
    if (addrDeps.empty())
      return true; // address is a pure constant/SSA computation

    // Whether op may write to any declared address dep. Specialized cases are
    // handled directly; every other side-effecting op falls through to a
    // generic memory-effects check so a write from e.g. fir.copy is not missed.
    auto mayWriteDeps = [&](mlir::Operation *op) -> bool {
      if (mlir::isMemoryEffectFree(op))
        return false;
      // hlfir.assign models a write to a box LHS as a Read of the descriptor,
      // so getModRef under-reports it; check the resolved destination directly.
      if (auto assign = mlir::dyn_cast<hlfir::AssignOp>(op)) {
        for (mlir::Value dep : addrDeps)
          if (!aliasAnalysis.alias(assign.getLhs(), dep).isNo())
            return true;
        return false;
      }
      // Region carriers (including fir.do_loop, which is opaque to getModRef)
      // are traversed by the walk; apply the canonical policy to leaf ops.
      // getModRef fails closed for an opaque leaf and unresolved writes.
      if (op->getNumRegions() != 0)
        return false;
      for (mlir::Value dep : addrDeps)
        if (aliasAnalysis.getModRef(op, dep).isMod())
          return true;
      return false;
    };

    mlir::WalkResult result = loop.walk([&](mlir::Operation *op) {
      return mayWriteDeps(op) ? mlir::WalkResult::interrupt()
                              : mlir::WalkResult::advance();
    });
    return !result.wasInterrupted();
  }

  /// P2 (interval): return true when nothing between the dominating init and
  /// the loop accesses the section. promote() folds the init away and
  /// overwrites the section with a post-loop copy-out, so a read or write there
  /// -- or a write to a subscript the address depends on -- would be
  /// miscompiled.
  bool sectionUntouchedBetweenInitAndLoop(hlfir::AssignOp init,
                                          fir::DoLoopOp loop, hlfir::Entity sec,
                                          llvm::ArrayRef<mlir::Value> addrDeps,
                                          mlir::DominanceInfo &domInfo,
                                          fir::AliasAnalysis &aliasAnalysis) {
    mlir::Operation *initOp = init.getOperation();
    mlir::Operation *loopOp = loop.getOperation();

    // Whether op itself accesses the section or writes one of its subscripts.
    auto touchesOp = [&](mlir::Operation *op) -> bool {
      if (mlir::isMemoryEffectFree(op))
        return false;
      // hlfir.assign models a write to a box LHS as a Read of the descriptor,
      // so getModRef under-reports it; check the resolved destination directly.
      if (auto asg = mlir::dyn_cast<hlfir::AssignOp>(op)) {
        if (!aliasAnalysis.alias(sec, asg.getLhs()).isNo())
          return true;
        // A direct copy `other = section` reads the section's original storage
        // through the RHS variable; promote() folds the init away and leaves
        // that storage stale. (An expr RHS reads memory only through inner ops,
        // which the walk visits, so restrict this to a variable RHS.)
        mlir::Value rhs = asg.getRhs();
        if (hlfir::Entity(rhs).isVariable() &&
            !aliasAnalysis.alias(sec, rhs).isNo())
          return true;
        for (mlir::Value dep : addrDeps)
          if (!aliasAnalysis.alias(asg.getLhs(), dep).isNo())
            return true;
        return false;
      }
      // Region carriers (including fir.do_loop, which is opaque to getModRef)
      // are traversed by touchesTree; a leaf touches the section if it reads or
      // writes it, or writes a subscript. getModRef fails closed for an opaque
      // leaf and unresolved effects.
      if (op->getNumRegions() != 0)
        return false;
      if (aliasAnalysis.getModRef(op, sec).isModOrRef())
        return true;
      for (mlir::Value dep : addrDeps)
        if (aliasAnalysis.getModRef(op, dep).isMod())
          return true;
      return false;
    };

    auto touchesTree = [&](mlir::Operation *root) {
      bool found = false;
      root->walk([&](mlir::Operation *op) {
        if (!found && touchesOp(op))
          found = true;
      });
      return found;
    };

    // Common case: the init and the loop are sequential in one block (the init
    // dominates the loop, so it precedes it). The interval is then exactly the
    // ops between them -- scan them linearly, skipping the whole-scope walk and
    // per-op dominance queries of the general path below.
    if (initOp->getBlock() == loopOp->getBlock()) {
      for (mlir::Operation *op = initOp->getNextNode(); op && op != loopOp;
           op = op->getNextNode())
        if (touchesTree(op))
          return false;
      return true;
    }

    mlir::Operation *scope = initOp->getParentOp();
    if (!scope)
      return true;
    bool clear = true;
    scope->walk([&](mlir::Operation *op) {
      if (!clear || op == initOp || op == loopOp)
        return;
      // "Between" = dominated by the init and not strictly after the loop, so
      // conditionally executed accesses in the init->loop window are included.
      if (domInfo.properlyDominates(initOp, op) &&
          !domInfo.properlyDominates(loopOp, op) && touchesTree(op))
        clear = false;
    });
    return clear;
  }

  /// P3 (writes): return true when nothing inside \p loop other than \p rmw may
  /// clobber the promoted section \p sec. Any surviving in-loop write to the
  /// section -- an hlfir.assign, a direct fir.array_coor+fir.store, or a call
  /// -- would be silently overwritten by the post-loop copy-out of T, so it
  /// blocks promotion. Writes are found via memory effects, failing closed for
  /// an unresolved write or an opaque side-effecting leaf.
  bool noAliasingWritesInLoop(fir::DoLoopOp loop, hlfir::AssignOp rmw,
                              hlfir::Entity sec,
                              fir::AliasAnalysis &aliasAnalysis) {
    mlir::WalkResult result = loop.walk([&](mlir::Operation *op) {
      if (op == rmw.getOperation() || mlir::isMemoryEffectFree(op))
        return mlir::WalkResult::advance();

      // hlfir.assign models a write to a box LHS as a Read of the descriptor,
      // so getModRef under-reports it; check the resolved destination directly.
      if (auto other = mlir::dyn_cast<hlfir::AssignOp>(op)) {
        if (!aliasAnalysis.alias(sec, other.getLhs()).isNo())
          return mlir::WalkResult::interrupt();
        return mlir::WalkResult::advance();
      }
      // A call's read of the section cannot be redirected to the temporary, so
      // reject a call that modifies OR references it (not just writes).
      if (mlir::isa<fir::CallOp, fir::DispatchOp>(op)) {
        if (aliasAnalysis.getModRef(op, sec).isModOrRef())
          return mlir::WalkResult::interrupt();
        return mlir::WalkResult::advance();
      }

      // Region carriers (including fir.do_loop, which is opaque to getModRef)
      // are traversed by the walk; a leaf that may write the section clobbers
      // it. This catches direct writes such as fir.array_coor + fir.store, and
      // getModRef fails closed for an opaque leaf and unresolved writes.
      if (op->getNumRegions() != 0)
        return mlir::WalkResult::advance();
      if (aliasAnalysis.getModRef(op, sec).isMod())
        return mlir::WalkResult::interrupt();
      return mlir::WalkResult::advance();
    });
    return !result.wasInterrupted();
  }

  /// P3 (reads): return true when every in-loop read that may alias the
  /// promoted section \p sec flows through a designator the rewrite redirects
  /// to the temporary (the RMW LHS \p lhsDes or a collected read, or one built
  /// on them). Any other aliasing read -- an element read a(k,i), an
  /// overlapping slice, or a direct fir.load / fir.copy / fir.array_load of the
  /// section -- would read stale memory once promoted.
  bool allAliasingAccessesRedirected(fir::DoLoopOp loop, hlfir::Entity sec,
                                     hlfir::DesignateOp lhsDes,
                                     llvm::ArrayRef<hlfir::DesignateOp> reads,
                                     fir::AliasAnalysis &aliasAnalysis) {
    llvm::SmallPtrSet<mlir::Operation *, 8> redirected;
    redirected.insert(lhsDes.getOperation());
    for (hlfir::DesignateOp d : reads)
      redirected.insert(d.getOperation());

    // A designator is safe if it or a designator it is built on is redirected;
    // replaceUsesWithIf rewires the whole chain to the temporary.
    auto derivesFromRedirected = [&](hlfir::DesignateOp d) {
      mlir::Value base = d.getResult();
      while (auto des = base.getDefiningOp<hlfir::DesignateOp>()) {
        if (redirected.contains(des.getOperation()))
          return true;
        base = des.getMemref();
      }
      return false;
    };
    // A read of \p v is safe when v flows through a redirected designator.
    auto readsRedirected = [&](mlir::Value v) {
      auto d = v.getDefiningOp<hlfir::DesignateOp>();
      return d && derivesFromRedirected(d);
    };

    mlir::WalkResult result = loop.walk([&](mlir::Operation *op) {
      // A section-aliasing designator must be one the rewrite redirects to T.
      if (auto d = mlir::dyn_cast<hlfir::DesignateOp>(op)) {
        if (!derivesFromRedirected(d) &&
            !aliasAnalysis.alias(sec, d.getResult()).isNo())
          return mlir::WalkResult::interrupt();
        return mlir::WalkResult::advance();
      }
      if (mlir::isMemoryEffectFree(op))
        return mlir::WalkResult::advance();
      // Any other op: a read that may alias the section must flow through a
      // redirected designator (fir.load, fir.copy, fir.array_load, ...) or it
      // would observe stale memory once the reduction accumulates into T.
      if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
        llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> effects;
        iface.getEffects(effects);
        for (const mlir::MemoryEffects::EffectInstance &e : effects) {
          if (!mlir::isa<mlir::MemoryEffects::Read>(e.getEffect()))
            continue;
          mlir::Value v = e.getValue();
          if (v && readsRedirected(v))
            continue;
          // An unresolved read (no value) could read anything: fail closed.
          if (!v || !aliasAnalysis.alias(sec, v).isNo())
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      }
      // No memory-effect interface: safe only if it is a region carrier whose
      // body is walked separately; an opaque leaf might read anything.
      if (op->getNumRegions() == 0)
        return mlir::WalkResult::interrupt();
      return mlir::WalkResult::advance();
    });
    return !result.wasInterrupted();
  }

  /// P4: total element count of the RHS when it has a compile-time-constant
  /// shape, or nullopt for a scalar or dynamic-extent RHS.
  static std::optional<int64_t> rhsConstantExtent(mlir::Value rhs) {
    auto exprTy = mlir::dyn_cast<hlfir::ExprType>(rhs.getType());
    if (!exprTy || exprTy.isScalar())
      return std::nullopt;
    int64_t count = 1;
    for (int64_t e : exprTy.getShape()) {
      // Overflow means an enormous shape, which the extent cap rejects anyway.
      if (mlir::ShapedType::isDynamic(e) || llvm::MulOverflow(count, e, count))
        return std::nullopt;
    }
    return count;
  }
};

} // namespace
