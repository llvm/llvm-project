//===- LiftSCFWhileToSCFFor.cpp - Rewrite counted scf.while as scf.for ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Walks each scf.while produced by lift-cf-to-scf and, when the canonical
// shape is recognized, rewrites it as an scf.for.
//
// Recognized shape:
//
//   scf.while (%trip = %tripInit, %iv = %ivInit, %x = %xInit, ...)
//             : (i32, i32, T, ...) -> (i32, i32, T, ...) {
//     %cmp = arith.cmpi sgt, %trip, %c0 : i32
//     %r:M = scf.if %cmp -> (i32, i32, T, ..., i32) {
//       // ...body...
//       %tripNext = arith.subi %trip, %c1     : i32
//       %ivNext   = arith.addi %iv,   %ivStep : i32   // or arith.subi
//       %xNext    = <any expr>
//       scf.yield %tripNext, %ivNext, %xNext, ..., %c1_marker : ...
//     } else {
//       scf.yield %poison, %poison, %poison, ..., %c0_marker : ...
//     }
//     %enter = arith.trunci %r#M-1 : i32 to i1
//     scf.condition(%enter) %r#0, %r#1, %r#2, ... : ...
//   } do {
//   ^bb0(%a0, %a1, %a2, ...):
//     scf.yield %a0, %a1, %a2, ... : ...           // identity
//   }
//
// Rewrite (canonical case: tripStep = -1, ivStep > 0):
//
//   %N        = %tripInit
//   %scaledN  = arith.muli %N, %ivStep : i32        // skipped when ivStep == 1
//   %ub       = arith.addi %ivInit, %scaledN : i32  // or %ivInit + %N
//   scf.for %i = %ivInit to %ub step %ivStep iter_args(%x = %xInit) -> (T) {
//     // body cloned from the scf.if continues branch, with
//     //   %iv -> %i, %x -> iter-arg, %trip unused
//     scf.yield %xNext : T
//   }
//
// Bailouts (no rewrite, just a stderr diagnostic):
//   - any result of the scf.while is used
//   - the scf.condition predicate doesn't trace to an scf.if's condition (we
//     need the if to extract the body from its continues branch)
//   - more than one induction-variable candidate (in addition to trip)
//   - zero induction-variable candidates
//   - ivStep is not positive (scf.for requires step > 0)
//   - trip step != -1 (N derivation would be wrong)
//   - the trip arg is referenced anywhere in the body except in the
//     recurrence and the gating cmp
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/raw_ostream.h"

namespace fir {
#define GEN_PASS_DEF_LIFTSCFWHILETOSCFFOR
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "lift-scf-while-to-scf-for"

namespace {
using namespace fir;
using namespace mlir;

// True if `x` is loop-invariant w.r.t. `whileOp` — its definition lives
// strictly outside the op.
static bool isLoopInvariant(Value x, scf::WhileOp whileOp) {
  if (Operation *def = x.getDefiningOp()) {
    return !whileOp->isProperAncestor(def);
  }
  Block *owner = cast<BlockArgument>(x).getOwner();
  Operation *ownerOp = owner->getParentOp();
  if (ownerOp == whileOp.getOperation()) {
    return false;
  }
  return !whileOp->isProperAncestor(ownerOp);
}

// Returns the value that, in the next iteration of `whileOp`, becomes the
// before-block argument at `argIdx`.
static Value getNextIterValueInBefore(scf::WhileOp whileOp, unsigned argIdx) {
  auto yieldOp = cast<scf::YieldOp>(whileOp.getAfterBody()->getTerminator());
  Value yielded = yieldOp.getOperand(argIdx);

  // This is the canonical case since the after region just passes through
  // values computed in the before region.
  if (auto afterArg = dyn_cast<BlockArgument>(yielded)) {
    if (afterArg.getOwner() == whileOp.getAfterBody()) {
      auto condOp = whileOp.getConditionOp();
      return condOp.getArgs()[afterArg.getArgNumber()];
    }
  }

  return yielded;
}

struct AffineRecurrence {
  // `op` is null if we failed to find an affine recurrence for a value `v`.
  Operation *op = nullptr;
  Value step;
  bool isSub = false;
  explicit operator bool() const { return op != nullptr; }
};

// If `v` is `arg + c` or `arg - c` (with `arg` matching `beforeArg` and `c`
// loop-invariant w.r.t. `whileOp`), return the recurrence op, the step value,
// and whether the recurrence is a subtraction. The returned step is the raw
// RHS; for arith.subi the caller should interpret it as a negative step.
static AffineRecurrence matchAffineRecurrence(Value v, Value beforeArg,
                                              scf::WhileOp whileOp) {
  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    if (add.getLhs() == beforeArg && isLoopInvariant(add.getRhs(), whileOp)) {
      return {add.getOperation(), add.getRhs(), /*isSub=*/false};
    }
    if (add.getRhs() == beforeArg && isLoopInvariant(add.getLhs(), whileOp)) {
      return {add.getOperation(), add.getLhs(), /*isSub=*/false};
    }
  }

  if (auto sub = v.getDefiningOp<arith::SubIOp>()) {
    if (sub.getLhs() == beforeArg && isLoopInvariant(sub.getRhs(), whileOp)) {
      return {sub.getOperation(), sub.getRhs(), /*isSub=*/true};
    }
  }

  return {};
}

// Locate the "continues" branch of an scf.if whose result feeds scf.condition.
// We use the heuristic: the structurizer encodes "loop continues" as an extra
// i32 result that gets trunc'd to the scf.condition predicate. The branch
// whose yield at that result index is a non-zero constant is the continues
// branch.
static Block *getContinuesBranch(scf::IfOp ifOp, Value condPredicate) {
  auto trunc = condPredicate.getDefiningOp<arith::TruncIOp>();
  if (!trunc) {
    return ifOp.thenBlock();
  }

  Value src = trunc.getIn();
  auto opRes = dyn_cast<OpResult>(src);
  if (!opRes || opRes.getOwner() != ifOp.getOperation()) {
    return ifOp.thenBlock();
  }

  unsigned predIdx = opRes.getResultNumber();

  auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
  auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());

  auto isNonZeroConst = [](Value v) {
    IntegerAttr attr;
    return matchPattern(v, m_Constant(&attr)) && attr.getInt() != 0;
  };

  if (isNonZeroConst(thenYield.getOperand(predIdx))) {
    return ifOp.thenBlock();
  }

  if (isNonZeroConst(elseYield.getOperand(predIdx))) {
    return ifOp.elseBlock();
  }

  return ifOp.thenBlock();
}

//===----------------------------------------------------------------------===//
// LoopInfo
//===----------------------------------------------------------------------===//

// Everything we need to drive the scf.while → scf.for rewrite when the
// canonical shape is recognized.
struct LoopInfo {
  scf::WhileOp whileOp;

  // The continues branch of the scf.if that gates the loop. Holds the body
  // we'll clone into the new scf.for.
  Block *continuesBlock = nullptr;

  // Trip counter.
  Value tripInit;
  Operation *tripRecurrence = nullptr;

  // Sole induction variable (becomes the scf.for IV).
  unsigned ivArgIdx = 0;
  Value ivInit; // lb
  Value ivStep; // signed, must be positive (we reject subi for now)
  Operation *ivRecurrence = nullptr;

  // Other before-args that pass through as iter_args of the new scf.for.
  // For each:
  //   - `argIdx`         : index in whileOp.getBeforeArguments(); also the
  //                        index into whileOp.getInits() that gives the
  //                        initial value of the iter_arg.
  //   - `contYieldSlot`  : operand index in the continues-branch yield that
  //                        carries this iter_arg's next-iteration value.
  //                        Equivalently, the index of the corresponding
  //                        scf.if result.
  // These are stored separately because the before-arg lane and the
  // condition/scf.if-result lane are independent in scf.while: their counts
  // and permutations need not match.
  struct IterArg {
    unsigned argIdx;
    unsigned contYieldSlot;
  };
  SmallVector<IterArg> iterArgs;

  LoopInfo(scf::WhileOp whileOp, Block *continuesBlock, Value tripInit,
           Operation *tripRecurrence, unsigned ivArgIdx, Value ivInit,
           Value ivStep, Operation *ivRecurrence, SmallVector<IterArg> iterArgs)
      : whileOp(whileOp), continuesBlock(continuesBlock), tripInit(tripInit),
        tripRecurrence(tripRecurrence), ivArgIdx(ivArgIdx), ivInit(ivInit),
        ivStep(ivStep), ivRecurrence(ivRecurrence),
        iterArgs(std::move(iterArgs)) {}
};

// Try to build a LoopInfo for `whileOp`. On failure, emit a one-line stderr
// diagnostic explaining why and return std::nullopt.
static std::optional<LoopInfo> tryBuildLoopInfo(scf::WhileOp whileOp,
                                                llvm::raw_ostream &os) {
  auto bail = [&](StringRef reason) -> std::optional<LoopInfo> {
    os << "  [skip] " << reason << "\n";
    return std::nullopt;
  };

  // Results of the scf.while must be unused.
  for (Value r : whileOp.getResults()) {
    if (!r.use_empty()) {
      return bail("scf.while result has uses");
    }
  }

  // The condition predicate must come from an scf.if whose continues branch
  // we can use as the loop body.
  Value condPredicate = whileOp.getConditionOp().getCondition();
  auto trunc = condPredicate.getDefiningOp<arith::TruncIOp>();

  // TODO: Maybe we can relax that in the future.
  if (!trunc) {
    return bail("scf.condition predicate is not arith.trunci");
  }

  auto ifOp = trunc.getIn().getDefiningOp<scf::IfOp>();

  if (!ifOp) {
    return bail("scf.condition predicate does not trace to an scf.if");
  }

  // The scf.if's predicate is the comparison that gates loop continuation.
  arith::CmpIOp cmp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();

  // TODO: Maybe we can make that more generic in the future.
  if (!cmp) {
    return bail("scf.if condition is not arith.cmpi");
  }

  // Find the IV-side of the gating cmp.
  bool lhsInv = isLoopInvariant(cmp.getLhs(), whileOp);
  bool rhsInv = isLoopInvariant(cmp.getRhs(), whileOp);
  Value cmpIV;

  if (rhsInv && !lhsInv) {
    cmpIV = cmp.getLhs();
  } else if (lhsInv && !rhsInv) {
    cmpIV = cmp.getRhs();
  } else {
    return bail("gating compare has no invariant side");
  }

  // Classify each before-arg as trip / IV / iter_arg.
  std::optional<unsigned> tripCounterArgIdx;
  AffineRecurrence tripCounterRecurrence;
  SmallVector<std::pair<unsigned, AffineRecurrence>> nonTripCounterRecurrences;
  SmallVector<LoopInfo::IterArg> iterArgs;

  Block *cont = getContinuesBranch(ifOp, condPredicate);
  auto contYield = cast<scf::YieldOp>(cont->getTerminator());

  for (auto [argIdx, beforeArg] :
       llvm::enumerate(whileOp.getBeforeArguments())) {
    Value nextInBefore = getNextIterValueInBefore(whileOp, (unsigned)argIdx);

    // In the canonical lift-cf-to-scf shape every iteration value flows
    // through one of `ifOp`'s results. We need the result index — both to
    // match the recurrence in the continues branch (for the trip + IV) and
    // to find the next-iteration value for pass-through iter_args.
    auto opRes = dyn_cast<OpResult>(nextInBefore);
    if (!opRes || opRes.getOwner() != ifOp.getOperation()) {
      return bail("before-arg's next-iter value does not come from the "
                  "predicate scf.if");
    }
    unsigned contYieldSlot = opRes.getResultNumber();
    Value branchSource = contYield.getOperand(contYieldSlot);
    AffineRecurrence rec =
        matchAffineRecurrence(branchSource, beforeArg, whileOp);

    if (beforeArg == cmpIV) {
      if (!rec) {
        return bail("gating cmp's IV operand has no affine recurrence");
      }

      if (tripCounterArgIdx) {
        return bail("multiple before-args match the gating cmp's IV operand");
      }

      tripCounterArgIdx = (unsigned)argIdx;
      tripCounterRecurrence = rec;
      continue;
    }

    if (rec) {
      nonTripCounterRecurrences.push_back({(unsigned)argIdx, rec});
    } else {
      iterArgs.push_back({(unsigned)argIdx, contYieldSlot});
    }
  }

  if (!tripCounterArgIdx) {
    return bail("no before-arg matches the gating cmp's IV operand");
  }

  // Trip step must be -1 (canonical lift output).
  IntegerAttr tripStepAttr;
  bool tripStepIsOne =
      matchPattern(tripCounterRecurrence.step, m_Constant(&tripStepAttr)) &&
      tripStepAttr.getInt() == 1;

  if (!tripCounterRecurrence.isSub || !tripStepIsOne) {
    return bail("trip-counter step is not -1");
  }

  // Exactly one non-trip recurrence is the induction variable. Other
  // recurrence-shaped before-args could in principle be supported as
  // affine-update iter_args, but that's not yet implemented.
  if (nonTripCounterRecurrences.empty()) {
    return bail("no induction variable candidate found");
  }

  if (nonTripCounterRecurrences.size() > 1) {
    return bail("multiple induction variable candidates (not yet supported)");
  }

  auto [ivIdx, ivRec] = nonTripCounterRecurrences.front();
  if (ivRec.isSub) {
    return bail("induction variable step is negative (not yet supported)");
  }

  LoopInfo info(whileOp, cont, whileOp.getInits()[*tripCounterArgIdx],
                tripCounterRecurrence.op, ivIdx, whileOp.getInits()[ivIdx],
                ivRec.step, ivRec.op, std::move(iterArgs));

  // The trip before-arg may only be used by the gating cmp and its own
  // recurrence. Any other use would have semantics we can't preserve after
  // dropping the trip counter.
  Value tripCounterArg = whileOp.getBeforeArguments()[*tripCounterArgIdx];

  for (OpOperand &use : tripCounterArg.getUses()) {
    Operation *user = use.getOwner();

    if (user == cmp.getOperation() || user == tripCounterRecurrence.op) {
      continue;
    }

    return bail("trip-counter before-arg has uses outside the cmp/recurrence");
  }

  // The trip recurrence's result must be used only by the scf.yield that
  // propagates it back to the next iteration. Anything else would require us
  // to materialize a trip-equivalent expression in the new scf.for, which we
  // don't support.
  for (OpOperand &use : tripCounterRecurrence.op->getResult(0).getUses()) {
    if (!isa<scf::YieldOp>(use.getOwner())) {
      return bail("trip-counter recurrence result has non-yield uses");
    }
  }

  return info;
}

//===----------------------------------------------------------------------===//
// Rewrite
//===----------------------------------------------------------------------===//

static void rewriteToSCFFor(const LoopInfo &info, llvm::raw_ostream &os) {
  scf::WhileOp whileOp = info.whileOp;
  OpBuilder builder(whileOp);
  Location loc = whileOp.getLoc();

  // The original IV is typed at whatever the source-level induction variable
  // used (commonly i32 for Fortran). Downstream affine / index-based
  // conversions expect index-typed scf.for bounds, so we always emit an
  // index-typed scf.for. The original IV type is restored via
  // arith.index_cast inside the body.
  Type origType = info.ivInit.getType();
  Type indexType = builder.getIndexType();
  auto castTo = [&](Value v, Type target) -> Value {
    if (v.getType() == target) {
      return v;
    }
    return arith::IndexCastOp::create(builder, loc, target, v);
  };

  // 1. Materialize ub_excl in the original type just before the scf.while,
  //    then cast lb/ub/step to index.
  //
  //   N      = tripInit
  //   scaled = (ivStep == 1) ? N : arith.muli N, ivStep
  //   ub     = arith.addi ivInit, scaled
  IntegerAttr ivStepAttr;
  bool ivStepIsOne = matchPattern(info.ivStep, m_Constant(&ivStepAttr)) &&
                     ivStepAttr.getInt() == 1;
  Value scaled =
      ivStepIsOne
          ? info.tripInit
          : arith::MulIOp::create(builder, loc, info.tripInit, info.ivStep);
  Value ub = arith::AddIOp::create(builder, loc, info.ivInit, scaled);

  Value lbIdx = castTo(info.ivInit, indexType);
  Value ubIdx = castTo(ub, indexType);
  Value stepIdx = castTo(info.ivStep, indexType);

  // 2. Collect iter_arg inits (from the iter_args we're passing through) and
  //    the corresponding "next" values (yielded inside the continues branch).
  //    The before-arg lane and the continues-yield lane are independent in
  //    scf.while, so we use the per-iter-arg `contYieldSlot` recorded during
  //    analysis to index `contYield` rather than the before-arg index.
  auto contYield = cast<scf::YieldOp>(info.continuesBlock->getTerminator());
  SmallVector<Value> iterInits;
  SmallVector<Value> iterNexts;
  iterInits.reserve(info.iterArgs.size());
  iterNexts.reserve(info.iterArgs.size());
  for (const LoopInfo::IterArg &ia : info.iterArgs) {
    iterInits.push_back(whileOp.getInits()[ia.argIdx]);
    iterNexts.push_back(contYield.getOperand(ia.contYieldSlot));
  }

  // 3. Create the scf.for with index-typed bounds. The default ForOp builder
  //    only auto-inserts a `scf.yield` terminator when `iterInits` is empty;
  //    for the non-empty case it expects the caller to either provide a body
  //    builder or insert the terminator itself. Insert a placeholder yield
  //    (operands populated in step 6) so the rest of the rewrite can position
  //    its inserts via `forBody->getTerminator()` uniformly.
  auto forOp =
      scf::ForOp::create(builder, loc, lbIdx, ubIdx, stepIdx, iterInits);

  Block *forBody = forOp.getBody();
  if (forBody->empty() || !forBody->back().hasTrait<OpTrait::IsTerminator>()) {
    OpBuilder termBuilder(builder.getContext());
    termBuilder.setInsertionPointToEnd(forBody);
    scf::YieldOp::create(termBuilder, loc, ValueRange{});
  }
  builder.setInsertionPoint(forBody->getTerminator());

  // 4. Build the value mapping:
  //    - Original IV before-arg → arith.index_cast of the scf.for IV (back to
  //      the original IV's type) so body ops keep working unchanged.
  //    - Original iter_args before-args → scf.for iter_arg block args.
  //    - Original trip before-arg → unused (validated to have no body uses).
  //    - Original trip recurrence result → unused (validated above).
  //
  //    The IV recurrence is left in the body-cloning loop below: SSA puts
  //    it before any user, the `ivArg → ivAsOrig` mapping handles its IV
  //    operand, and the loop-invariant step passes through as-is. Cloning
  //    rather than re-creating preserves attributes (nsw, nuw, ...) and
  //    works whether the original was arith.addi or arith.subi.
  IRMapping mapping;
  Value ivAsOrig = castTo(forOp.getInductionVar(), origType);

  mapping.map(whileOp.getBeforeArguments()[info.ivArgIdx], ivAsOrig);
  for (auto [i, ia] : llvm::enumerate(info.iterArgs)) {
    mapping.map(whileOp.getBeforeArguments()[ia.argIdx],
                forBody->getArgument(1 + i));
  }

  // 5. Clone every op from the continues block into the scf.for body, except:
  //    - the trip recurrence (its operand %trip has no mapping; cloning it
  //      would leave a dangling reference to the erased before-block arg)
  //    - the scf.yield terminator (we emit our own below)
  for (Operation &op : *info.continuesBlock) {
    if (&op == info.tripRecurrence) {
      continue;
    }
    if (isa<scf::YieldOp>(op)) {
      continue;
    }
    builder.clone(op, mapping);
  }

  // 6. Update the scf.for's existing terminator with the remapped next-values
  //    for the iter_args.
  SmallVector<Value> yieldOperands;
  yieldOperands.reserve(iterNexts.size());
  for (Value v : iterNexts) {
    yieldOperands.push_back(mapping.lookupOrDefault(v));
  }
  forBody->getTerminator()->setOperands(yieldOperands);

  // 7. The scf.while's results (trip exit value, iv exit value, iter_arg exit
  //    values) are all unused by precondition. We can simply erase the while.
  whileOp.erase();

  os << "  [rewritten] scf.while → scf.for at ";
  loc.print(os);
  os << "\n";
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

class LiftSCFWhileToSCFFor
    : public fir::impl::LiftSCFWhileToSCFForBase<LiftSCFWhileToSCFFor> {
public:
  using LiftSCFWhileToSCFForBase<
      LiftSCFWhileToSCFFor>::LiftSCFWhileToSCFForBase;

  void runOnOperation() override {
    auto &os = llvm::errs();
    // Collect first; rewrite separately so we don't mutate during walk.
    SmallVector<scf::WhileOp> worklist;
    getOperation()->walk([&](scf::WhileOp w) { worklist.push_back(w); });

    os << "\n[lift-scf-while-to-scf-for] discovered " << worklist.size()
       << " scf.while op(s) in " << getOperation()->getName() << "\n";

    unsigned rewritten = 0;
    for (scf::WhileOp whileOp : worklist) {
      os << "\n[lift-scf-while-to-scf-for] scf.while at ";
      whileOp.getLoc().print(os);
      os << "\n";

      std::optional<LoopInfo> info = tryBuildLoopInfo(whileOp, os);
      if (!info) {
        continue;
      }
      rewriteToSCFFor(*info, os);
      ++rewritten;
    }

    if (!worklist.empty()) {
      unsigned skipped = worklist.size() - rewritten;
      os << "\n[lift-scf-while-to-scf-for] summary: " << rewritten
         << " rewritten, " << skipped << " skipped (of " << worklist.size()
         << " discovered)\n";
    }
  }
};

} // namespace
