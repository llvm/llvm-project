//===- CIRFunctionExitAnalysis.cpp - How control leaves a function --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A reachability walk over structured CIR. Each region reports how control
// leaves it (falling through, break, continue, explicit return, or reaching an
// `implicit` terminator), and each structured operation combines the exits of
// its regions according to its semantics. Reachable noreturn calls, user
// traps and throws are one more kind of exit, the edges CheckFallThrough uses
// to tell MaybeFallThrough from AlwaysFallThrough.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/CIRFunctionExitAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"
#include "clang/CIR/Dialect/Analysis/CIRConstantFolding.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/TimeProfiler.h"

using namespace cir;

namespace {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

/// How control can leave a region or a structured operation.
enum class Exits : uint8_t {
  None = 0,
  Normal = 1 << 0,
  Break = 1 << 1,
  Continue = 1 << 2,
  ExplicitReturn = 1 << 3,
  ImplicitReturn = 1 << 4,
  // A reachable noreturn call, user trap, throw or `__builtin_unreachable`:
  // the edges CheckFallThrough uses to tell MaybeFallThrough from
  // AlwaysFallThrough.
  NoReturn = 1 << 5,
  // Outcomes of a reachable `cir.condition` terminating the region.
  CondTrue = 1 << 6,
  CondFalse = 1 << 7,
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/CondFalse)
};

/// The exits that leave the whole function.
constexpr Exits functionExits =
    Exits::ExplicitReturn | Exits::ImplicitReturn | Exits::NoReturn;

/// The exits that leave an enclosing operation too, not just the region.
constexpr Exits escapingExits = Exits::Break | Exits::Continue | functionExits;

/// Whether control may leave through any of `kinds`.
bool mayExitVia(Exits exits, Exits kinds) {
  return (exits & kinds) != Exits::None;
}

/// The labels gotos can reach. A reached goto may target a label in a region
/// a walk saw as dead, so the function is walked again until no new label
/// becomes a target.
class GotoTargets {
public:
  explicit GotoTargets(FuncOp func) : func(func) {}

  /// Forget the gotos the previous walk reached.
  void startWalk() {
    reachedGotos.clear();
    reachedIndirectGoto = false;
  }
  void addReachedGoto(GotoOp op) { reachedGotos.insert(op.getLabelAttr()); }
  /// A cir.indirect_goto can reach every label whose address is taken.
  void addReachedIndirectGoto() { reachedIndirectGoto = true; }
  /// Add the labels the walk's gotos target; whether any of them is new.
  bool finishWalk();

  bool empty() const { return targets.empty(); }
  bool isTarget(mlir::Operation *op) const {
    auto label = mlir::dyn_cast<LabelOp>(op);
    return label && targets.contains(label.getLabelAttr());
  }
  /// Whether a target is nested in `op`, so a dead `op` still needs a visit.
  bool enclosesTarget(mlir::Operation *op) const {
    return targetAncestors.contains(op);
  }

private:
  void addAddressTakenLabels();
  void updateTargetAncestors();

  FuncOp func;
  llvm::DenseSet<mlir::StringAttr> targets;
  llvm::DenseSet<mlir::Operation *> targetAncestors;
  llvm::DenseSet<mlir::StringAttr> reachedGotos;
  bool reachedIndirectGoto = false;
  bool addedAddressTakenLabels = false;
  /// The function's labels, collected once a goto makes them matter. Never
  /// empty after that, since a goto only targets a label of this function.
  llvm::SmallVector<LabelOp> labels;
};

class FunctionExitAnalyzer {
public:
  explicit FunctionExitAnalyzer(FuncOp func) : func(func), gotoTargets(func) {}

  FunctionExitKind run();

  /// Whether `op` may complete normally when control enters it at its start.
  /// Only exact if no goto can enter `op` through a label inside it.
  bool mayCompleteWhenEntered(mlir::Operation *op) {
    return mayExitVia(visitStructured(op, /*entered=*/true), Exits::Normal);
  }

private:
  Exits visitRegion(mlir::Region &region, bool entered);
  Exits visitStructured(mlir::Operation *op, bool entered);
  Exits visitLoop(LoopOpInterface loop, bool entered);
  Exits visitSwitch(SwitchOp op, bool entered);
  Exits visitCleanupScope(CleanupScopeOp op, bool entered);
  void visitTerminator(mlir::Operation *op, Exits &exits,
                       llvm::SmallVectorImpl<mlir::Block *> &liveSuccessors);

  FuncOp func;
  GotoTargets gotoTargets;
  /// The enclosing switches, innermost last. A reachable switch dispatches to
  /// all of its cases, wherever they are nested in its body.
  struct SwitchState {
    bool entered = false;
    bool hasDefault = false;
  };
  llvm::SmallVector<SwitchState> switches;

  /// Makes a switch the innermost one while it is in scope.
  class SwitchScope {
  public:
    SwitchScope(FunctionExitAnalyzer &analyzer, bool entered)
        : switches(analyzer.switches) {
      switches.push_back({entered, /*hasDefault=*/false});
    }
    SwitchScope(const SwitchScope &) = delete;
    SwitchScope &operator=(const SwitchScope &) = delete;
    ~SwitchScope() { switches.pop_back(); }

    /// Whether a default case was dispatched to so far.
    bool hasDefault() const { return switches.back().hasDefault; }

  private:
    llvm::SmallVector<SwitchState> &switches;
  };

  bool inEnteredSwitch() const {
    return !switches.empty() && switches.back().entered;
  }
};

} // namespace

/// The outcomes `cond` may have, as `CondTrue` and `CondFalse`: both, unless it
/// folds to a constant.
static Exits getCondExits(mlir::Value cond) {
  std::optional<bool> value = tryFoldConstantBool(cond);
  if (!value)
    return Exits::CondTrue | Exits::CondFalse;
  return *value ? Exits::CondTrue : Exits::CondFalse;
}

/// `__builtin_assume(0)` is noreturn for the CFG check.
static bool isAssumeFalse(mlir::Operation *op) {
  auto assume = mlir::dyn_cast<AssumeOp>(op);
  return assume && tryFoldConstantBool(assume.getPredicate()) == false;
}

/// The only operation with regions in a single-block \p region, if any.
static mlir::Operation *getOnlyRegionOp(mlir::Region &region) {
  if (!region.hasOneBlock())
    return nullptr;
  mlir::Operation *found = nullptr;
  for (mlir::Operation &op : region.front()) {
    if (op.getNumRegions() == 0)
      continue;
    if (found)
      return nullptr;
    found = &op;
  }
  return found;
}

/// The code written in a coroutine's body, without the implicit `try` (and the
/// `if` around it) CIRGen adds with exceptions: leaving those is not reaching
/// the end of the body.
// TODO: Drop this once CIRGen emits that handling in the coroutine's planned
// `unwind` region, leaving only the user's code in `body`.
static mlir::Region &getCoroutineUserBody(CoroutineOp coro) {
  mlir::Operation *op = getOnlyRegionOp(coro.getBody());
  if (auto ifOp = mlir::dyn_cast_if_present<IfOp>(op);
      ifOp && ifOp.getElseRegion().empty())
    op = getOnlyRegionOp(ifOp.getThenRegion());
  if (auto scope = mlir::dyn_cast_if_present<ScopeOp>(op))
    op = getOnlyRegionOp(scope.getScopeRegion());
  if (auto tryOp = mlir::dyn_cast_if_present<TryOp>(op))
    return tryOp.getTryRegion();
  return coro.getBody();
}

FunctionExitKind FunctionExitAnalyzer::run() {
  // For a coroutine, the end is that of the code written in its body, and
  // co_return is its return.
  mlir::Region *body = &func.getBody();
  if (func.getCoroutineKind().has_value()) {
    CoroutineOp coro;
    func.walk<mlir::WalkOrder::PreOrder>([&](CoroutineOp op) {
      coro = op;
      return mlir::WalkResult::interrupt();
    });
    if (!coro)
      return FunctionExitKind::Unknown;
    body = &getCoroutineUserBody(coro);
  }

  // Without gotos, one walk.
  Exits exits;
  do {
    gotoTargets.startWalk();
    exits = visitRegion(*body, /*entered=*/true);
  } while (gotoTargets.finishWalk());

  bool explicitReturn = mayExitVia(exits, Exits::ExplicitReturn);
  if (!mayExitVia(exits, Exits::Normal | Exits::ImplicitReturn))
    return explicitReturn ? FunctionExitKind::ReturnOnly
                          : FunctionExitKind::NoReturn;
  if (explicitReturn || mayExitVia(exits, Exits::NoReturn))
    return FunctionExitKind::MayReachEnd;
  return FunctionExitKind::MustReachEnd;
}

bool GotoTargets::finishWalk() {
  size_t numTargets = targets.size();
  targets.insert_range(reachedGotos);
  if (reachedIndirectGoto)
    addAddressTakenLabels();
  if (targets.size() == numTargets)
    return false;
  updateTargetAncestors();
  return true;
}

void GotoTargets::addAddressTakenLabels() {
  if (addedAddressTakenLabels)
    return;
  addedAddressTakenLabels = true;
  mlir::StringAttr funcName = func.getSymNameAttr();
  auto addFromOp = [&](mlir::Operation *op) {
    walkBlockAddressLabels(
        op, [&](mlir::StringAttr labelFunc, mlir::StringAttr label) {
          if (labelFunc == funcName)
            targets.insert(label);
        });
  };
  // As in the CFG, addresses taken in static initializers count too.
  func.getBody().walk(addFromOp);
  if (auto module = func->getParentOfType<mlir::ModuleOp>())
    for (GlobalOp global : module.getOps<GlobalOp>())
      global->walk(addFromOp);
}

void GotoTargets::updateTargetAncestors() {
  if (labels.empty())
    func.getBody().walk([&](LabelOp label) { labels.push_back(label); });
  targetAncestors.clear();
  for (LabelOp label : labels) {
    if (!targets.contains(label.getLabelAttr()))
      continue;
    for (mlir::Operation *parent = label->getParentOp(); parent != func;
         parent = parent->getParentOp())
      if (!targetAncestors.insert(parent).second)
        break;
  }
}

Exits FunctionExitAnalyzer::visitRegion(mlir::Region &region, bool entered) {
  if (region.empty())
    return entered ? Exits::Normal : Exits::None;

  // Dead code is still walked when a goto or a switch dispatch may enter it,
  // and walked again if it turns out live.
  bool mayHideCase = inEnteredSwitch();
  auto mustVisitDead = [&](mlir::Operation &op) {
    return (mayHideCase && op.getNumRegions() != 0) ||
           gotoTargets.enclosesTarget(&op);
  };
  // Most regions have a single block, which an inline map holds without
  // allocating.
  Exits exits = Exits::None;
  llvm::SmallDenseMap<mlir::Block *, bool, 4> visited;
  llvm::SmallVector<std::pair<mlir::Block *, bool>> worklist;
  worklist.push_back({&region.front(), entered});
  if (!region.hasOneBlock() && (mayHideCase || !gotoTargets.empty()))
    for (mlir::Block &block : llvm::drop_begin(region))
      if (llvm::any_of(block, [&](mlir::Operation &op) {
            return gotoTargets.isTarget(&op) || mustVisitDead(op);
          }))
        worklist.push_back({&block, false});

  while (!worklist.empty()) {
    auto [block, live] = worklist.pop_back_val();
    auto [it, inserted] = visited.try_emplace(block, live);
    if (!inserted) {
      if (it->second || !live)
        continue;
      it->second = true;
    }

    // Querying the terminator trait once per block rather than per operation
    // is measurably faster.
    mlir::Operation *terminator =
        block->mightHaveTerminator() ? &block->back() : nullptr;
    llvm::SmallVector<mlir::Block *> liveSuccessors;
    for (mlir::Operation &op : *block) {
      if (!live) {
        if (gotoTargets.isTarget(&op)) {
          live = true;
          continue;
        }
        if (!mustVisitDead(op))
          continue;
      }

      if (op.getNumRegions() != 0) {
        Exits opExits = visitStructured(&op, live);
        exits |= opExits & escapingExits;
        live = mayExitVia(opExits, Exits::Normal);
        continue;
      }

      if (!live)
        continue;

      auto call = mlir::dyn_cast<CallOp>(op);
      if ((call && call.isNoReturn()) || isAssumeFalse(&op) ||
          mlir::isa<ThrowOp>(op)) {
        exits |= Exits::NoReturn;
        live = false;
      } else if (&op == terminator) {
        visitTerminator(&op, exits, liveSuccessors);
        live = false;
      }
    }

    // A block without a terminator falls through to the region's parent.
    if (live)
      exits |= Exits::Normal;

    for (mlir::Block *succ : liveSuccessors)
      worklist.push_back({succ, true});
  }
  return exits;
}

void FunctionExitAnalyzer::visitTerminator(
    mlir::Operation *op, Exits &exits,
    llvm::SmallVectorImpl<mlir::Block *> &liveSuccessors) {
  if (mlir::isa<YieldOp>(op)) {
    exits |= Exits::Normal;
  } else if (auto condOp = mlir::dyn_cast<ConditionOp>(op)) {
    exits |= getCondExits(condOp.getCondition());
  } else if (mlir::isa<CoReturnOp>(op)) {
    exits |= Exits::ExplicitReturn;
  } else if (auto synthesized =
                 mlir::dyn_cast<SynthesizedTerminatorOpInterface>(op)) {
    // User traps and __builtin_unreachable are noreturn, like in the CFG.
    if (synthesized.isImplicit())
      exits |= Exits::ImplicitReturn;
    else if (mlir::isa<ReturnOp>(op))
      exits |= Exits::ExplicitReturn;
    else
      exits |= Exits::NoReturn;
  } else if (mlir::isa<BreakOp>(op)) {
    exits |= Exits::Break;
  } else if (mlir::isa<ContinueOp>(op)) {
    exits |= Exits::Continue;
  } else if (auto brCond = mlir::dyn_cast<BrCondOp>(op)) {
    Exits outcomes = getCondExits(brCond.getCond());
    if (mayExitVia(outcomes, Exits::CondTrue))
      liveSuccessors.push_back(brCond.getDestTrue());
    if (mayExitVia(outcomes, Exits::CondFalse))
      liveSuccessors.push_back(brCond.getDestFalse());
  } else if (auto gotoOp = mlir::dyn_cast<GotoOp>(op)) {
    gotoTargets.addReachedGoto(gotoOp);
  } else if (mlir::isa<IndirectGotoOp>(op)) {
    gotoTargets.addReachedIndirectGoto();
  } else {
    // Everything else with successors (cir.br, cir.switch.flat, ...)
    // reaches all of them.
    llvm::append_range(liveSuccessors, op->getSuccessors());
  }
}

Exits FunctionExitAnalyzer::visitStructured(mlir::Operation *op, bool entered) {
  if (auto loop = mlir::dyn_cast<LoopOpInterface>(op))
    return visitLoop(loop, entered);
  if (auto switchOp = mlir::dyn_cast<SwitchOp>(op))
    return visitSwitch(switchOp, entered);
  if (auto cleanupScope = mlir::dyn_cast<CleanupScopeOp>(op))
    return visitCleanupScope(cleanupScope, entered);

  if (auto caseOp = mlir::dyn_cast<CaseOp>(op)) {
    // A case is entered by falling through from the previous one or by the
    // switch dispatching to it.
    bool dispatched = inEnteredSwitch();
    if (dispatched && caseOp.getKind() == CaseOpKind::Default)
      switches.back().hasDefault = true;
    return visitRegion(caseOp.getCaseRegion(), entered || dispatched);
  }

  if (auto ifOp = mlir::dyn_cast<IfOp>(op)) {
    Exits outcomes = getCondExits(ifOp.getCondition());
    bool thenEntered = entered && mayExitVia(outcomes, Exits::CondTrue);
    bool elseEntered = entered && mayExitVia(outcomes, Exits::CondFalse);
    Exits thenExits = visitRegion(ifOp.getThenRegion(), thenEntered);
    Exits elseExits = visitRegion(ifOp.getElseRegion(), elseEntered);
    return (thenExits | elseExits) & (Exits::Normal | escapingExits);
  }

  if (auto tryOp = mlir::dyn_cast<TryOp>(op)) {
    // Like the CFG check, treat every handler as reachable when the try is.
    Exits exits = visitRegion(tryOp.getTryRegion(), entered);
    for (mlir::Region &handler : tryOp.getHandlerRegions())
      exits |= visitRegion(handler, entered);
    return exits & (Exits::Normal | escapingExits);
  }

  // Any region may run. Scopes and ternaries complete if a region does; ops
  // whose regions do not yield control complete whenever reached.
  bool yieldsControl = mlir::isa<ScopeOp, TernaryOp>(op);
  Exits exits = entered && !yieldsControl ? Exits::Normal : Exits::None;
  Exits kept = yieldsControl ? Exits::Normal | escapingExits : escapingExits;
  for (mlir::Region &region : op->getRegions())
    exits |= visitRegion(region, entered) & kept;
  return exits;
}

Exits FunctionExitAnalyzer::visitLoop(LoopOpInterface loop, bool entered) {
  bool isDoWhile = &loop.getEntry() == &loop.getBody();
  bool condEntered = entered && !isDoWhile;
  bool bodyEntered = entered && isDoWhile;
  mlir::Region *step = loop.maybeGetStep();

  // Visit the regions in execution order. Another round is only needed when
  // the loop was not entered from the top but through a goto into it.
  Exits condExits = Exits::None, bodyExits = Exits::None,
        stepExits = Exits::None;
  auto visitBodyAndStep = [&]() {
    bodyExits = visitRegion(loop.getBody(), bodyEntered);
    bool reachesLatch = mayExitVia(bodyExits, Exits::Normal | Exits::Continue);
    if (step) {
      stepExits = visitRegion(*step, reachesLatch);
      reachesLatch = mayExitVia(stepExits, Exits::Normal);
    }
    return reachesLatch;
  };
  while (true) {
    if (isDoWhile) {
      condEntered |= visitBodyAndStep();
      condExits = visitRegion(loop.getCond(), condEntered);
      if (!mayExitVia(condExits, Exits::CondTrue) || bodyEntered)
        break;
      bodyEntered = true;
    } else {
      condExits = visitRegion(loop.getCond(), condEntered);
      bodyEntered |= mayExitVia(condExits, Exits::CondTrue);
      if (!visitBodyAndStep() || condEntered)
        break;
      condEntered = true;
    }
  }

  // Breaks and continues stop at the loop; what leaves the function leaves it.
  Exits exits = (condExits | bodyExits | stepExits) & functionExits;
  if (mayExitVia(condExits, Exits::CondFalse) ||
      mayExitVia(bodyExits, Exits::Break))
    exits |= Exits::Normal;
  return exits;
}

Exits FunctionExitAnalyzer::visitSwitch(SwitchOp op, bool entered) {
  // The switch body is entered through its cases only, so code ahead of the
  // first case is dead.
  SwitchScope scope(*this, entered);
  Exits bodyExits = visitRegion(op.getBody(), /*entered=*/false);

  // Breaks stop at the switch; continues and what leaves the function leave
  // it.
  Exits exits = bodyExits & (Exits::Continue | functionExits);
  // As in the CFG check, a switch listing every enumerator has no implicit
  // default edge.
  if (mayExitVia(bodyExits, Exits::Normal | Exits::Break) ||
      (entered && !scope.hasDefault() && !op.getAllEnumCasesCovered()))
    exits |= Exits::Normal;
  return exits;
}

Exits FunctionExitAnalyzer::visitCleanupScope(CleanupScopeOp op, bool entered) {
  Exits bodyExits = visitRegion(op.getBodyRegion(), entered);
  if (op.getCleanupKind() == CleanupKind::EH)
    return bodyExits;

  // A cleanup that cannot complete (e.g. a noreturn destructor) stops every
  // path leaving the body.
  bool leavesBody = mayExitVia(bodyExits, Exits::Normal | escapingExits);
  Exits cleanupExits = visitRegion(op.getCleanupRegion(), leavesBody);
  if (!leavesBody || mayExitVia(cleanupExits, Exits::Normal))
    return bodyExits;
  return Exits::NoReturn;
}

llvm::StringRef cir::stringifyFunctionExitKind(FunctionExitKind kind) {
  switch (kind) {
  case FunctionExitKind::Unknown:
    return "unknown";
  case FunctionExitKind::ReturnOnly:
    return "return_only";
  case FunctionExitKind::NoReturn:
    return "no_return";
  case FunctionExitKind::MayReachEnd:
    return "may_reach_end";
  case FunctionExitKind::MustReachEnd:
    return "must_reach_end";
  }
  llvm_unreachable("unknown FunctionExitKind");
}

static bool mayComplete(FuncOp func, mlir::Operation *op);

/// Conservatively, whether control may get past `last` in `block` (the whole
/// block when null). Only the closest region-holding op before it matters.
static bool mayReachPast(FuncOp func, mlir::Block &block,
                         mlir::Operation *last) {
  for (mlir::Operation *op = last; op; op = op->getPrevNode()) {
    // A goto may enter here, past anything before the label.
    if (mlir::isa<LabelOp>(op))
      return true;
    if (op->getNumRegions() != 0)
      return mayComplete(func, op);
  }
  return block.isEntryBlock() || !block.hasNoPredecessors();
}

/// Conservatively, whether `op` may complete normally: a scope when a block of
/// its body may reach its end, anything else analyzed on its own unless it
/// holds a label.
static bool mayComplete(FuncOp func, mlir::Operation *op) {
  mlir::Region *body = nullptr;
  if (auto scope = mlir::dyn_cast<ScopeOp>(op))
    body = &scope.getScopeRegion();
  else if (auto cleanupScope = mlir::dyn_cast<CleanupScopeOp>(op))
    body = &cleanupScope.getBodyRegion();
  if (!body) {
    // A goto to a label inside `op` could revive code the analysis of `op`
    // alone sees as dead.
    bool hasLabel = op->walk([](LabelOp) {
                        return mlir::WalkResult::interrupt();
                      }).wasInterrupted();
    return hasLabel || FunctionExitAnalyzer(func).mayCompleteWhenEntered(op);
  }
  if (body->empty())
    return true;
  return llvm::any_of(*body, [&](mlir::Block &block) {
    if (block.empty())
      return mayReachPast(func, block, nullptr);
    if (!block.mightHaveTerminator())
      return mayReachPast(func, block, &block.back());
    mlir::Operation *terminator = &block.back();
    return mlir::isa<YieldOp>(terminator) &&
           mayReachPast(func, block, terminator->getPrevNode());
  });
}

bool cir::mayReachEnd(FuncOp func) {
  // A coroutine ends where the code in its body does, not at an implicit
  // terminator.
  if (func.getCoroutineKind().has_value())
    return true;
  // CIRGen emits the implicit terminator only in the function's top-level
  // region; it is reached if what precedes it may complete.
  return llvm::any_of(func.getBody(), [&](mlir::Block &block) {
    if (block.empty())
      return mayReachPast(func, block, nullptr);
    if (!block.mightHaveTerminator())
      return mayReachPast(func, block, &block.back());
    mlir::Operation *terminator = &block.back();
    auto synthesized =
        mlir::dyn_cast<SynthesizedTerminatorOpInterface>(terminator);
    return synthesized && synthesized.isImplicit() &&
           mayReachPast(func, block, terminator->getPrevNode());
  });
}

FunctionExitKind cir::analyzeFunctionExits(FuncOp func) {
  // Declarations and aliases have no body to analyze.
  if (func.getBody().empty())
    return FunctionExitKind::Unknown;
  FunctionExitKind kind = FunctionExitAnalyzer(func).run();
#ifdef EXPENSIVE_CHECKS
  assert(((kind != FunctionExitKind::MayReachEnd &&
           kind != FunctionExitKind::MustReachEnd) ||
          mayReachEnd(func)) &&
         "mayReachEnd rejected a function that reaches its end");
#endif
  return kind;
}

llvm::SmallVector<FunctionExitResult>
cir::analyzeFunctionExits(mlir::ModuleOp module,
                          llvm::function_ref<bool(FuncOp)> filter) {
  llvm::TimeTraceScope timeScope("CIRFunctionExitAnalysis");
  llvm::SmallVector<FuncOp> funcs;
  for (FuncOp func : module.getOps<FuncOp>())
    if (!func.getBody().empty())
      funcs.push_back(func);

  // Waking the thread pool costs about as much as analyzing a thousand
  // functions, so smaller modules stay on this thread. Otherwise each task
  // takes every `numTasks`-th function, spreading out costly ones; the filter
  // runs in the tasks too, as it costs about as much as the analysis.
  constexpr size_t minFuncsForThreads = 1024;
  constexpr size_t funcsPerTask = 64;
  size_t numTasks = funcs.size() < minFuncsForThreads
                        ? 1
                        : llvm::divideCeil(funcs.size(), funcsPerTask);
  llvm::SmallVector<std::optional<FunctionExitKind>> kinds(funcs.size());
  mlir::parallelFor(module.getContext(), 0, numTasks, [&](size_t task) {
    for (size_t i = task; i < funcs.size(); i += numTasks)
      if (!filter || filter(funcs[i]))
        kinds[i] = analyzeFunctionExits(funcs[i]);
  });

  llvm::SmallVector<FunctionExitResult> results;
  for (auto [func, kind] : llvm::zip_equal(funcs, kinds))
    if (kind)
      results.push_back({func, *kind});
  return results;
}
