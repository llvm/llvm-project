//===-- SimplifyDoLoop.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// General-purpose FIR loop canonicalization pass.
//
// Transforms fir.do_loop nests into a canonical form suitable for affine
// promotion and loop optimizations (tiling, fusion, interchange, etc.).
//
// The canonical form has:
//   - No iter_args (shadow induction variable copies removed)
//   - No memory-based IV tracking inside the loop body
//   - Final IV values computed and stored after the outermost loop
//
// === Design Overview ===
//
// Analysis phase (per loop nest):
//   1. Collect perfectly nested fir.do_loop chain.
//   2. For each loop, verify iter_arg is a shadow of the induction variable:
//      - init = fir.convert(lower_bound)
//      - yield = arith.addi(iter_arg_or_load_of_iv, fir.convert(step))
//   3. Verify safety conditions:
//      a. Only one store to IV alloca inside loop (the init store of iter_arg)
//      b. No function/subroutine calls in the nest
//      c. IV alloca does not escape (only load/store/declare users)
//      d. Loop results are only used for final IV stores
//
// Transformation phase:
//   1. For each loop (innermost first):
//      a. Remove the initial store (fir.store %iter_arg to %iv_alloca)
//      b. Forward all loads of IV alloca inside loop body to fir.convert(IV)
//      todo: the forwarding of load of iv alloca can be done by some other pass
//      like fir-memref-dataflow-opt pass (if it is available).
//      c. Strip iter_args and fir.result, rebuild as simple fir.do_loop
//   2. After the outermost loop, compute and store final IV values
//      for all loops whose IV is live after the loop (outer to inner order).
//      Fortran final value: final_iv = lb + ((ub - lb + step) / step) * step
//      which equals the value of the iter_arg after the last increment.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace fir {
#define GEN_PASS_DEF_SIMPLIFYDOLOOP
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "simplify-do-loop"

using namespace fir;
using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Per-loop bookkeeping built during analysis
//===----------------------------------------------------------------------===//

struct LoopIVInfo {
  fir::DoLoopOp loop;
  Value ivAlloca;                  // fir.alloca for this loop's IV
  SmallVector<Value, 2> ivAliases; // ivAlloca + any fir.declare alias
  Value lowerBound;                // index-typed lower bound
  Value upperBound;                // index-typed upper bound
  Value step;                      // index-typed step
  Type ivType;                     // Fortran IV type (e.g. i32)
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Collect the IV memory reference and all its aliases (the raw fir.alloca
/// and any fir.declare results that alias it).  `ivRef` may be either the
/// alloca itself or a declare result — we normalise to the underlying alloca
/// first, then collect all declare aliases from it.
static SmallVector<Value, 2> collectAliases(Value ivRef) {
  SmallVector<Value, 2> aliases;

  // If ivRef is a declare result, trace back to the underlying alloca.
  Value underlying = ivRef;
  if (auto decl = ivRef.getDefiningOp<fir::DeclareOp>())
    underlying = decl.getMemref();

  aliases.push_back(underlying);
  for (auto *user : underlying.getUsers())
    if (auto decl = dyn_cast<fir::DeclareOp>(user))
      aliases.push_back(decl.getResult());

  return aliases;
}

/// Collect a singly-nested chain of fir.do_loop ops starting from `outer`.
/// Each loop body must contain exactly one inner fir.do_loop; other operations
/// are permitted.  Safety checks (no calls, single IV store, IV doesn't escape)
/// are enforced later by analyzeNest().
static SmallVector<fir::DoLoopOp> collectNest(fir::DoLoopOp outer) {
  SmallVector<fir::DoLoopOp> nest;
  fir::DoLoopOp cur = outer;
  while (cur) {
    nest.push_back(cur);
    fir::DoLoopOp inner;
    unsigned loopCount = 0;
    for (auto &op : cur.getBody()->getOperations())
      if (auto nested = dyn_cast<fir::DoLoopOp>(op)) {
        inner = nested;
        ++loopCount;
      }
    if (loopCount != 1)
      break;
    cur = inner;
  }
  return nest;
}

/// Strip fir.convert chains to find the root SSA value.
static Value stripConverts(Value val) {
  while (auto conv = val.getDefiningOp<fir::ConvertOp>())
    val = conv.getValue();
  return val;
}

/// Check whether `val` originates from `target` (possibly through fir.convert).
static bool originatesFrom(Value val, Value target) {
  return stripConverts(val) == target;
}

/// Find IV alloca: the first fir.store in the loop body whose value
/// originates from the iter_arg or the induction variable (possibly through
/// fir.convert chains).
// ***** We scan the entire top-level body rather than
/// stopping at an inner fir.do_loop so that the pass remains robust if
/// upstream passes reorder operations.
static Value findIVAlloca(fir::DoLoopOp loop) {
  if (!loop.hasIterOperands() || loop.getNumIterOperands() < 1)
    return {};
  auto iterArg = loop.getRegionIterArgs()[0];
  auto iv = loop.getInductionVar();
  for (auto &op : loop.getBody()->getOperations()) {
    if (auto store = dyn_cast<fir::StoreOp>(op)) {
      Value stored = store.getValue();
      if (originatesFrom(stored, iterArg) || originatesFrom(stored, iv))
        return store.getMemref();
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
//                          ANALYSIS PHASE
//===----------------------------------------------------------------------===//

// ---- Analysis 1: Confirm iter_arg is a shadow of the induction variable ----
//
// The iter_arg must mirror the index-typed induction variable:
//   init  = fir.convert(lower_bound) : (index) -> i32
//   yield = arith.addi(iter_arg_or_load_of_iv, fir.convert(step))

static bool isShadowIV(fir::DoLoopOp loop, Value ivAlloca) {
  auto iterOperands = loop.getIterOperands();
  auto iterArg = loop.getRegionIterArgs()[0];

  auto initConvert = iterOperands[0].getDefiningOp<fir::ConvertOp>();
  if (!initConvert || initConvert.getValue() != loop.getLowerBound()) {
    LLVM_DEBUG(llvm::dbgs() << "  [shadow] init is not fir.convert(lb)\n");
    return false;
  }

  auto resultOp = cast<fir::ResultOp>(loop.getBody()->getTerminator());
  auto addOp = resultOp.getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!addOp) {
    LLVM_DEBUG(llvm::dbgs() << "  [shadow] yield is not arith.addi\n");
    return false;
  }

  auto isIVValue = [&](Value v) -> bool {
    if (v == iterArg)
      return true;
    if (auto load = v.getDefiningOp<fir::LoadOp>()) {
      if (load.getMemref() == ivAlloca)
        return true;
      if (auto decl = load.getMemref().getDefiningOp<fir::DeclareOp>())
        if (decl.getMemref() == ivAlloca)
          return true;
    }
    return false;
  };

  Value stepSide;
  if (isIVValue(addOp.getLhs()))
    stepSide = addOp.getRhs();
  else if (isIVValue(addOp.getRhs()))
    stepSide = addOp.getLhs();
  else {
    LLVM_DEBUG(llvm::dbgs() << "  [shadow] addi doesn't use iter_arg/IV\n");
    return false;
  }

  auto stepConvert = stepSide.getDefiningOp<fir::ConvertOp>();
  if (!stepConvert || stepConvert.getValue() != loop.getStep()) {
    LLVM_DEBUG(llvm::dbgs() << "  [shadow] step operand mismatch\n");
    return false;
  }
  return true;
}

// ---- Analysis 2: Only one store to IV alloca inside loop (the init store) --

static bool singleStoreToIVAlloca(fir::DoLoopOp loop,
                                  ArrayRef<Value> ivAliases) {
  auto iterArg = loop.getRegionIterArgs()[0];
  auto iv = loop.getInductionVar();
  bool foundInit = false;
  bool ok = true;

  loop.walk([&](fir::StoreOp store) {
    if (!llvm::is_contained(ivAliases, store.getMemref()))
      return;
    if (!foundInit && (originatesFrom(store.getValue(), iterArg) ||
                       originatesFrom(store.getValue(), iv))) {
      foundInit = true;
      return;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  [store] extra store to IV: " << store << "\n");
    ok = false;
  });
  return ok;
}

// ---- Analysis 3: No function/subroutine calls in the nest -----------------

static bool noCallsInNest(fir::DoLoopOp outermost) {
  bool ok = true;
  outermost.walk([&](Operation *op) {
    if (isa<fir::CallOp>(op) || isa<func::CallOp>(op) ||
        isa<fir::DispatchOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "  [call] found: " << *op << "\n");
      ok = false;
    }
  });
  return ok;
}

// ---- Analysis 4: IV alloca must not escape --------------------------------

static bool ivDoesNotEscape(ArrayRef<Value> ivAliases) {
  for (auto alias : ivAliases)
    for (auto *user : alias.getUsers()) {
      if (auto store = dyn_cast<fir::StoreOp>(user)) {
        if (store.getMemref() != alias) {
          LLVM_DEBUG(llvm::dbgs() << "  [escape] IV used as stored value: "
                                  << *user << "\n");
          return false;
        }
        continue;
      }
      if (!isa<fir::LoadOp, fir::DeclareOp>(user)) {
        LLVM_DEBUG(llvm::dbgs() << "  [escape] IV escapes: " << *user << "\n");
        return false;
      }
    }
  return true;
}

// ---- Check if a bound value can be safely rematerialized after the loop ---
// Runs during analysis (pre-transformation) to reject nests whose bounds
// contain ops that cannot be correctly duplicated after the outermost loop.
//
// Safe:  values defined outside the outermost loop, loop IVs (block args of
//        fir.do_loop — resolved via ivFinalMap), fir.convert, arith constants,
//        and arithmetic over safe values.  Loads of IV allocas are safe because
//        transformOneLoop will forward them to fir.convert(IV) before
//        rematerializeOutside runs.
// Unsafe: fir.load of a non-IV address inside the loop — the memory may have
//         been modified between the original load and the post-loop insertion
//         point, so duplicating the load would read a wrong value.

static bool canSafelyRematerialize(Value val, fir::DoLoopOp outermost,
                                   ArrayRef<LoopIVInfo> infos) {
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    auto *owner = blockArg.getOwner()->getParentOp();
    if (!outermost->isAncestor(owner))
      return true;
    return isa<fir::DoLoopOp>(owner);
  }

  auto *defOp = val.getDefiningOp();
  assert(defOp &&
         "expected value to be a block argument or have a defining op");
  if (!outermost->isAncestor(defOp))
    return true;

  if (auto conv = dyn_cast<fir::ConvertOp>(*defOp))
    return canSafelyRematerialize(conv.getValue(), outermost, infos);

  if (auto load = dyn_cast<fir::LoadOp>(*defOp)) {
    for (const auto &info : infos)
      if (llvm::is_contained(info.ivAliases, load.getMemref()))
        return true;
    LLVM_DEBUG(llvm::dbgs()
               << "  [remat] non-IV load in bound: " << *defOp << "\n");
    return false;
  }

  if (isa<arith::ConstantOp>(*defOp))
    return true;

  if (defOp->getNumResults() == 1 && mlir::isPure(defOp)) {
    for (Value operand : defOp->getOperands())
      if (!canSafelyRematerialize(operand, outermost, infos))
        return false;
    return true;
  }

  return false;
}

// ---- Full nest analysis ---------------------------------------------------

static bool analyzeNest(SmallVector<LoopIVInfo> &infos) {
  // --- Per-loop: shadow-IV check, IV alloca discovery, single-store check ---
  for (auto &info : infos) {
    auto loop = info.loop;
    if (!loop.hasIterOperands() || loop.getNumIterOperands() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "  skip: loop has != 1 iter_args at "
                              << loop.getLoc() << "\n");
      return false;
    }

    info.ivAlloca = findIVAlloca(loop);
    if (!info.ivAlloca) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  cannot find IV alloca at " << loop.getLoc() << "\n");
      return false;
    }

    info.ivAliases = collectAliases(info.ivAlloca);

    if (!isShadowIV(loop, info.ivAlloca)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  not shadow IV at " << loop.getLoc() << "\n");
      return false;
    }

    if (!singleStoreToIVAlloca(loop, info.ivAliases)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  multiple stores at " << loop.getLoc() << "\n");
      return false;
    }

    // Record loop bounds and IV type from the iter_arg init value.
    info.lowerBound = loop.getLowerBound();
    info.upperBound = loop.getUpperBound();
    info.step = loop.getStep();
    info.ivType = loop.getIterOperands()[0].getType();
  }

  // --- No function calls in the nest ---
  if (!noCallsInNest(infos.front().loop))
    return false;

  // --- IV alloca must not escape ---
  for (auto &info : infos) {
    if (!ivDoesNotEscape(info.ivAliases))
      return false;
  }

  // --- Loop results must only be used for final IV stores ---
  for (auto &info : infos) {
    for (auto result : info.loop.getResults()) {
      for (auto *user : result.getUsers()) {
        auto store = dyn_cast<fir::StoreOp>(user);
        if (!store || !llvm::is_contained(info.ivAliases, store.getMemref())) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  [result] loop result used outside IV store at "
                     << info.loop.getLoc() << ": " << *user << "\n");
          return false;
        }
      }
    }
  }

  // --- Verify that loop bounds can be safely rematerialized after the loop ---
  fir::DoLoopOp outermost = infos.front().loop;
  for (auto &info : infos) {
    if (!canSafelyRematerialize(info.lowerBound, outermost, infos) ||
        !canSafelyRematerialize(info.upperBound, outermost, infos) ||
        !canSafelyRematerialize(info.step, outermost, infos)) {
      LLVM_DEBUG(llvm::dbgs() << "  bounds not safely rematerializable at "
                              << info.loop.getLoc() << "\n");
      return false;
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
//                       TRANSFORMATION PHASE
//===----------------------------------------------------------------------===//

/// Ensure a value is available (dominates) at the current insertion point.
/// If the value is already defined outside `outermost`, return it directly.
/// Otherwise, rematerialize the computation by cloning through simple ops
/// (fir.convert, arith constants, arithmetic).
///
/// Precondition: canSafelyRematerialize() has already verified that the
/// bound values do not depend on non-IV loads inside the loop.  Any IV loads
/// (fir.load of IV alloca) have been forwarded to fir.convert(IV) by
/// transformOneLoop before this function is called.
///
/// `ivFinalMap` maps loop induction variables (block arguments) to their
/// already-computed final index values.  This allows inner loop bounds that
/// depend on outer IVs (e.g. triangular loops) to be correctly resolved.
static Value rematerializeOutside(Value val, fir::DoLoopOp outermost,
                                  OpBuilder &builder, Location loc,
                                  const DenseMap<Value, Value> &ivFinalMap) {
  // Already defined outside the outermost loop — use directly.
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    if (!outermost->isAncestor(blockArg.getOwner()->getParentOp()))
      return val;
    auto it = ivFinalMap.find(val);
    if (it != ivFinalMap.end())
      return it->second;
    return val;
  }
  auto *defOp = val.getDefiningOp();
  if (!defOp || !outermost->isAncestor(defOp))
    return val;

  // fir.convert: rematerialize the input, then re-emit the convert.
  if (auto conv = dyn_cast<fir::ConvertOp>(*defOp)) {
    Value newInput = rematerializeOutside(conv.getValue(), outermost, builder,
                                          loc, ivFinalMap);
    return fir::ConvertOp::create(builder, loc, conv.getType(), newInput);
  }

  // arith.constant: just clone it.
  if (isa<arith::ConstantOp>(*defOp)) {
    auto *cloned = builder.clone(*defOp);
    return cloned->getResult(0);
  }

  // Pure ops (no side effects): rematerialize all operands recursively,
  // then clone the op with new operands.
  if (defOp->getNumResults() == 1 && mlir::isPure(defOp)) {
    SmallVector<Value> newOperands;
    for (auto operand : defOp->getOperands())
      newOperands.push_back(
          rematerializeOutside(operand, outermost, builder, loc, ivFinalMap));
    auto *cloned = builder.clone(*defOp);
    for (unsigned i = 0; i < newOperands.size(); ++i)
      cloned->setOperand(i, newOperands[i]);
    return cloned->getResult(0);
  }

  return val;
}

/// Compute the Fortran final IV value and store it to the IV alloca.
///
/// Fortran DO loop semantics: after normal completion, the IV holds the
/// value it would have received on the iteration that causes termination.
/// For `DO I = lb, ub, step`:
///   trip_count = MAX((ub - lb + step) / step, 0)
///   final_iv   = lb + trip_count * step
///
/// Since the loop actually executed (we wouldn't reach here otherwise for
/// an empty nest), we use the FIR loop's own bounds which are already
/// index-typed. We compute:
///   final_index = lb + ((ub - lb + step) / step) * step
/// Then convert to the Fortran IV type (e.g. i32) and store.
///
/// `ivFinalMap` is populated with the mapping from this loop's IV (block arg)
/// to its *last iteration value* (finalIndex - step).  Inner loops whose
/// bounds depend on an outer IV need the value from the last iteration, not
/// the Fortran final value (which is one step past the last iteration).
/// Example: for `do i=1,100; do j=1,i`, j's final value must be computed
/// with i=100 (last iteration), not i=101 (Fortran final).
static void emitFinalIVStore(OpBuilder &builder, Location loc, LoopIVInfo &info,
                             fir::DoLoopOp outermost,
                             DenseMap<Value, Value> &ivFinalMap) {
  // Rematerialize bounds outside the outermost loop if needed.
  // For inner loops with IV-dependent bounds (e.g. do j=1,i), the outer IV
  // block argument will be resolved via ivFinalMap.
  Value lb = rematerializeOutside(info.lowerBound, outermost, builder, loc,
                                  ivFinalMap);
  Value ub = rematerializeOutside(info.upperBound, outermost, builder, loc,
                                  ivFinalMap);
  Value step =
      rematerializeOutside(info.step, outermost, builder, loc, ivFinalMap);

  // trip_count = (ub - lb + step) / step
  Value ubMinusLb = arith::SubIOp::create(builder, loc, ub, lb);
  Value ubMinusLbPlusStep =
      arith::AddIOp::create(builder, loc, ubMinusLb, step);
  Value tripCount =
      arith::DivSIOp::create(builder, loc, ubMinusLbPlusStep, step);

  // Clamp trip count to >= 0.
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value isPositive = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::sgt, tripCount, zero);
  Value clampedTrip =
      arith::SelectOp::create(builder, loc, isPositive, tripCount, zero);

  // final_index = lb + trip_count * step
  Value tripTimesStep = arith::MulIOp::create(builder, loc, clampedTrip, step);
  Value finalIndex = arith::AddIOp::create(builder, loc, lb, tripTimesStep);

  // Record the *last iteration* value (finalIndex - step) for this IV so
  // that inner loops whose bounds depend on this IV use the correct value.
  // Fortran final value = lb + trip_count * step (one step PAST the last
  // iteration), but the inner loop's last execution sees the outer IV at
  // lb + (trip_count - 1) * step.
  Value lastIterValue = arith::SubIOp::create(builder, loc, finalIndex, step);
  ivFinalMap[info.loop.getInductionVar()] = lastIterValue;

  // Convert from index to the Fortran IV type (e.g. i32).
  Value finalIV = fir::ConvertOp::create(builder, loc, info.ivType, finalIndex);

  // Store to the IV alloca.
  fir::StoreOp::create(builder, loc, finalIV, info.ivAlloca);

  LLVM_DEBUG(llvm::dbgs() << "  emitted final IV store for " << info.ivAlloca
                          << " at " << loc << "\n");
}

/// Transform one loop: remove init/final stores, forward IV loads, strip
/// iter_args, and rebuild as a simple fir.do_loop.
static fir::DoLoopOp transformOneLoop(fir::DoLoopOp loop,
                                      ArrayRef<Value> ivAliases,
                                      OpBuilder &builder) {
  auto loc = loop.getLoc();
  auto iv = loop.getInductionVar();
  auto iterArg = loop.getRegionIterArgs()[0];

  LLVM_DEBUG(llvm::dbgs() << "  transforming loop at " << loc << "\n");

  // Identify the increment addi (yielded by fir.result).
  auto resultOp = cast<fir::ResultOp>(loop.getBody()->getTerminator());
  Operation *incrementOp = nullptr;
  if (auto addOp = resultOp.getOperand(0).getDefiningOp<arith::AddIOp>())
    incrementOp = addOp;

  // --- Remove initial store to IV alloca ---
  // The init store may be:  fir.store %iterArg to %alloca
  //                    or:  fir.store (fir.convert %iterArg) to %alloca
  //                    or:  fir.store (fir.convert %iv) to %alloca
  // Scan the loop body (before any inner loop) and erase the first store
  // to any IV alias whose value originates from iterArg or the IV.
  for (auto &op : llvm::make_early_inc_range(*loop.getBody())) {
    if (auto store = dyn_cast<fir::StoreOp>(op)) {
      if (llvm::is_contained(ivAliases, store.getMemref()) &&
          (originatesFrom(store.getValue(), iterArg) ||
           originatesFrom(store.getValue(), iv))) {
        // Any dead fir.convert chain feeding this store will be cleaned up
        // by the subsequent canonicalize pass in the pipeline.
        store.erase();
        break; // only remove the first (init) store
      }
    }
  }

  // --- Remove final store: fir.store %loop_result to %iv_alloca ---
  for (auto result : loop.getResults()) {
    for (auto *user : llvm::make_early_inc_range(result.getUsers()))
      if (auto store = dyn_cast<fir::StoreOp>(user))
        if (llvm::is_contained(ivAliases, store.getMemref()))
          store.erase();
  }

  // --- Forward loads of IV alloca anywhere inside loop → fir.convert(IV) ---
  // The initial store was removed, so loads of the IV alloca inside the
  // loop (including nested loops) now need to read from the index-typed
  // induction variable (converted to the IV's Fortran type).
  loop.walk([&](fir::LoadOp load) {
    if (llvm::is_contained(ivAliases, load.getMemref())) {
      builder.setInsertionPoint(load);
      auto ivCast = fir::ConvertOp::create(builder, loc, load.getType(), iv);
      load.getResult().replaceAllUsesWith(ivCast);
      load.erase();
    }
  });

  // --- Replace remaining iter_arg uses with fir.convert(IV) ---
  {
    SmallVector<OpOperand *> uses;
    for (auto &use : iterArg.getUses())
      uses.push_back(&use);

    for (auto *use : uses) {
      if (use->getOwner() == incrementOp)
        continue;
      builder.setInsertionPoint(use->getOwner());
      auto ivCast = fir::ConvertOp::create(builder, loc, iterArg.getType(), iv);
      use->set(ivCast);
    }
  }

  // --- Clear fir.result operands ---
  auto *terminator = loop.getBody()->getTerminator();
  terminator->eraseOperands(0, terminator->getNumOperands());

  // Erase the increment addi (its result was the fir.result operand).
  if (incrementOp && incrementOp->use_empty())
    incrementOp->erase();

  // --- Rebuild loop without iter_args ---
  // Preserve `unordered` and `loopAnnotation`.  `finalValue` is dropped —
  // the final IV is now handled by an explicit post-loop store.
  builder.setInsertionPoint(loop);
  auto newLoop = fir::DoLoopOp::create(
      builder, loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      /*unordered=*/loop.getUnordered().has_value(),
      /*finalCountValue=*/false,
      /*iterArgs=*/mlir::ValueRange{});
  if (auto annotation = loop.getLoopAnnotationAttr())
    newLoop.setLoopAnnotationAttr(annotation);
  loop.getInductionVar().replaceAllUsesWith(newLoop.getInductionVar());

  auto &oldOps = loop.getBody()->getOperations();
  auto &newOps = newLoop.getBody()->getOperations();
  newOps.splice(newOps.begin(), oldOps, oldOps.begin(),
                std::prev(oldOps.end()));

  loop.erase();
  return newLoop;
}

//===----------------------------------------------------------------------===//
// Pass entry
//===----------------------------------------------------------------------===//

class SimplifyDoLoop : public fir::impl::SimplifyDoLoopBase<SimplifyDoLoop> {
public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    // Collect all outermost fir.do_loop ops.
    SmallVector<fir::DoLoopOp> outerLoops;
    func.walk([&](fir::DoLoopOp loop) {
      if (!loop->getParentOfType<fir::DoLoopOp>())
        outerLoops.push_back(loop);
    });

    for (fir::DoLoopOp outerLoop : outerLoops) {
      SmallVector<fir::DoLoopOp> nestLoops = collectNest(outerLoop);
      LLVM_DEBUG(llvm::dbgs()
                 << "SimplifyDoLoop: nest depth " << nestLoops.size() << " at "
                 << outerLoop.getLoc() << "\n");

      // ======== Analysis Phase ========
      SmallVector<LoopIVInfo> infos;
      for (fir::DoLoopOp loop : nestLoops)
        infos.push_back({loop, {}, {}, {}, {}, {}, {}});

      if (!analyzeNest(infos)) {
        LLVM_DEBUG(llvm::dbgs() << "  nest rejected by analysis\n");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "  analysis passed — transforming "
                              << infos.size() << " loops\n");

      // ======== Transformation Phase ========
      OpBuilder builder(func.getContext());

      for (int i = infos.size() - 1; i >= 0; --i)
        infos[i].loop =
            transformOneLoop(infos[i].loop, infos[i].ivAliases, builder);

      // ---- After the outermost loop, emit final IV value stores. ----
      //         Process outer-to-inner so that outer IV final values are
      //         available when computing inner IV finals (e.g. triangular
      //         loops where inner bounds depend on outer IVs).
      fir::DoLoopOp outermostNew = infos.front().loop;
      builder.setInsertionPointAfter(outermostNew);

      DenseMap<Value, Value> ivFinalMap;
      for (auto &info : infos)
        emitFinalIVStore(builder, outermostNew.getLoc(), info, outermostNew,
                         ivFinalMap);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::createSimplifyDoLoopPass() {
  return std::make_unique<SimplifyDoLoop>();
}
