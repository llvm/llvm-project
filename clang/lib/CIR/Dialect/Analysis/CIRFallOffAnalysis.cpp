//===- CIRFallOffAnalysis.cpp - Can control flow off the end of a function ===//
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
// traps and throws are recorded as the abnormal and fake edges Sema's CFG
// check uses to tell "maybe" from "always".
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/CIRFallOffAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Threading.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/TimeProfiler.h"

using namespace cir;

namespace {

/// How control can leave a region or a structured operation.
struct Exits {
  bool normal = false;
  bool breaks = false;
  bool continues = false;
  bool explicitReturn = false;
  bool implicitReturn = false;
  // Outcomes of a reachable `cir.condition` terminating the region.
  bool condTrue = false;
  bool condFalse = false;

  /// Merge the exits that escape an enclosed operation.
  void mergeEscaping(const Exits &other) {
    breaks |= other.breaks;
    continues |= other.continues;
    explicitReturn |= other.explicitReturn;
    implicitReturn |= other.implicitReturn;
  }

  bool any() const {
    return normal || breaks || continues || explicitReturn || implicitReturn;
  }
};

class FallOffAnalyzer {
public:
  explicit FallOffAnalyzer(FuncOp func, bool tailOnly = false)
      : func(func), tailOnly(tailOnly) {}

  FallOffKind run();

  /// Conservatively, whether `op` alone completes when entered at the top. A
  /// label inside it, or dead code that may hold one, means yes.
  bool mayCompleteFromTop(mlir::Operation *op) {
    assert(tailOnly && "needs a tail-only analyzer");
    return visitStructured(op, /*entered=*/true).normal || unsure;
  }

private:
  Exits visitRegion(mlir::Region &region, bool entered);
  Exits visitStructured(mlir::Operation *op, bool entered);
  Exits visitLoop(LoopOpInterface loop, bool entered);
  Exits visitSwitch(SwitchOp op, bool entered);
  Exits visitCleanupScope(CleanupScopeOp op, bool entered);
  void visitTerminator(mlir::Operation *op, Exits &exits,
                       llvm::SmallVectorImpl<mlir::Block *> &liveSuccessors);
  void addAddressTakenLabels();
  void updateLabelAncestors();

  /// Whether a dead operation still needs a visit because a label inside it
  /// can be reached through a goto.
  bool containsTargetedLabel(mlir::Operation *op) const {
    return labelAncestors.contains(op);
  }
  bool isTargetedLabel(mlir::Operation *op) const {
    auto label = mlir::dyn_cast<LabelOp>(op);
    return label && targetedLabels.contains(label.getLabelAttr());
  }

  FuncOp func;
  bool tailOnly;
  bool unsure = false;
  bool abnormalEdge = false;
  bool fakeEdge = false;
  /// Labels control can reach through a goto or an indirect branch, and the
  /// operations enclosing them.
  llvm::DenseSet<mlir::StringAttr> targetedLabels;
  llvm::DenseSet<mlir::Operation *> labelAncestors;
  /// Labels targeted by the gotos reached so far.
  llvm::DenseSet<mlir::StringAttr> reachedGotoTargets;
  /// Whether a cir.indirect_goto is reachable; it can reach every label whose
  /// address is taken.
  bool reachedIndirectGoto = false;
  bool addedAddressTakenLabels = false;
  /// The function's labels, collected once a goto makes them matter.
  std::optional<llvm::SmallVector<LabelOp>> labels;
  /// The enclosing switches, innermost last. A reachable switch dispatches to
  /// all of its cases, wherever they are nested in its body.
  struct SwitchState {
    bool entered = false;
    bool hasDefault = false;
  };
  llvm::SmallVector<SwitchState> switches;

  bool inEnteredSwitch() const {
    return !switches.empty() && switches.back().entered;
  }
};

} // namespace

/// Fold `value` to a constant through side-effect-free operations.
static mlir::Attribute foldToConstant(mlir::Value value, unsigned depth = 0) {
  mlir::Attribute attr;
  if (mlir::matchPattern(value, mlir::m_Constant(&attr)))
    return attr;

  constexpr unsigned maxDepth = 8;
  if (depth >= maxDepth)
    return {};

  // C conditions such as `while (1)` or `x || 1` convert between int and
  // bool, which the cast folder leaves alone.
  if (auto cast = value.getDefiningOp<CastOp>()) {
    if (cast.getKind() == CastKind::int_to_bool) {
      if (auto intAttr = mlir::dyn_cast_if_present<IntAttr>(
              foldToConstant(cast.getSrc(), depth + 1)))
        return BoolAttr::get(value.getContext(), !intAttr.getValue().isZero());
      return {};
    }
    if (cast.getKind() == CastKind::bool_to_int) {
      if (auto boolAttr = mlir::dyn_cast_if_present<BoolAttr>(
              foldToConstant(cast.getSrc(), depth + 1)))
        return IntAttr::get(cast.getType(), boolAttr.getValue() ? 1 : 0);
      return {};
    }
  }

  // `x || 1` and `x && 0` become a cir.ternary whose arms yield the same
  // constant, which the CFG check folds too.
  if (auto ternary = value.getDefiningOp<TernaryOp>()) {
    auto foldArm = [&](mlir::Region &arm) -> mlir::Attribute {
      if (!arm.hasOneBlock() || !arm.front().mightHaveTerminator())
        return {};
      auto yield = mlir::dyn_cast<YieldOp>(arm.front().getTerminator());
      if (!yield || yield.getNumOperands() != 1)
        return {};
      return foldToConstant(yield.getOperand(0), depth + 1);
    };
    mlir::Attribute trueValue = foldArm(ternary.getTrueRegion());
    mlir::Attribute falseValue = foldArm(ternary.getFalseRegion());
    if (auto cond = mlir::dyn_cast_if_present<BoolAttr>(
            foldToConstant(ternary.getCond(), depth + 1)))
      return cond.getValue() ? trueValue : falseValue;
    return trueValue == falseValue ? trueValue : mlir::Attribute();
  }

  mlir::Operation *def = value.getDefiningOp();
  if (!def || def->getNumResults() != 1 || def->getNumRegions() != 0 ||
      !mlir::isMemoryEffectFree(def))
    return {};
  llvm::SmallVector<mlir::Attribute> operands;
  for (mlir::Value operand : def->getOperands()) {
    operands.push_back(foldToConstant(operand, depth + 1));
    if (!operands.back())
      return {};
  }
  llvm::SmallVector<mlir::OpFoldResult> results;
  if (mlir::failed(def->fold(operands, results)) || results.size() != 1)
    return {};
  return mlir::dyn_cast_if_present<mlir::Attribute>(results.front());
}

static std::optional<bool> getConstantBool(mlir::Value value) {
  mlir::Attribute attr = foldToConstant(value);
  if (auto boolAttr = mlir::dyn_cast_if_present<BoolAttr>(attr))
    return boolAttr.getValue();
  return std::nullopt;
}

static bool isNoReturnCall(mlir::Operation *op) {
  return mlir::isa<CallOp>(op) &&
         op->hasAttr(CIRDialect::getNoReturnAttrName());
}

/// `__builtin_assume(0)` is noreturn for the CFG check.
static bool isAssumeFalse(mlir::Operation *op) {
  auto assume = mlir::dyn_cast<AssumeOp>(op);
  return assume && getConstantBool(assume.getPredicate()) == false;
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
/// `if` around it) CIRGen adds with exceptions: leaving those is not fall-off.
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

FallOffKind FallOffAnalyzer::run() {
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
      return FallOffKind::Unknown;
    body = &getCoroutineUserBody(coro);
  }

  // A reached goto may target a label in a region already walked as dead, so
  // walk again until no new label is targeted; without gotos, once.
  Exits exits;
  while (true) {
    abnormalEdge = fakeEdge = reachedIndirectGoto = false;
    reachedGotoTargets.clear();
    exits = visitRegion(*body, /*entered=*/true);

    size_t numTargeted = targetedLabels.size();
    targetedLabels.insert_range(reachedGotoTargets);
    if (reachedIndirectGoto)
      addAddressTakenLabels();
    if (targetedLabels.size() == numTargeted)
      break;
    updateLabelAncestors();
  }

  bool plainEdge = exits.implicitReturn || exits.normal;
  if (!plainEdge)
    return exits.explicitReturn ? FallOffKind::Never
                                : FallOffKind::NeverAndNoReturn;
  if (abnormalEdge || fakeEdge || exits.explicitReturn)
    return FallOffKind::Maybe;
  return FallOffKind::Always;
}

void FallOffAnalyzer::addAddressTakenLabels() {
  if (addedAddressTakenLabels)
    return;
  addedAddressTakenLabels = true;
  mlir::StringAttr funcName = func.getSymNameAttr();
  auto addFromAttr = [&](mlir::Attribute attr) {
    attr.walk(
        [&](BlockAddrInfoAttr info) {
          if (info.getFunc().getAttr() == funcName)
            targetedLabels.insert(info.getLabel());
        },
        [&](BlockAddrDiffAttr diff) {
          if (diff.getFunc().getAttr() == funcName) {
            targetedLabels.insert(diff.getLhsLabel());
            targetedLabels.insert(diff.getRhsLabel());
          }
        });
  };
  auto addFromOp = [&](mlir::Operation *op) {
    if (auto blockAddr = mlir::dyn_cast<BlockAddressOp>(op))
      targetedLabels.insert(blockAddr.getBlockAddrInfo().getLabel());
    else if (mlir::isa<ConstantOp, GlobalOp>(op))
      for (mlir::NamedAttribute attr : op->getAttrs())
        addFromAttr(attr.getValue());
  };
  // As in the CFG, addresses taken in static initializers count too.
  func.getBody().walk(addFromOp);
  if (auto module = func->getParentOfType<mlir::ModuleOp>())
    for (GlobalOp global : module.getOps<GlobalOp>())
      global->walk(addFromOp);
}

void FallOffAnalyzer::updateLabelAncestors() {
  if (!labels) {
    labels.emplace();
    func.getBody().walk([&](LabelOp label) { labels->push_back(label); });
  }
  labelAncestors.clear();
  for (LabelOp label : *labels) {
    if (!targetedLabels.contains(label.getLabelAttr()))
      continue;
    for (mlir::Operation *parent = label->getParentOp(); parent != func;
         parent = parent->getParentOp())
      if (!labelAncestors.insert(parent).second)
        break;
  }
}

Exits FallOffAnalyzer::visitRegion(mlir::Region &region, bool entered) {
  Exits exits;
  if (region.empty()) {
    exits.normal = entered;
    return exits;
  }

  // Dead code is still walked when a goto or a switch dispatch may enter it,
  // and walked again if it turns out live.
  bool mayHideCase = inEnteredSwitch();
  auto mustVisitDead = [&](mlir::Operation &op) {
    return (mayHideCase && op.getNumRegions() != 0) ||
           containsTargetedLabel(&op);
  };
  // Most regions have a single block, which an inline map holds without
  // allocating.
  llvm::SmallDenseMap<mlir::Block *, bool, 4> visited;
  llvm::SmallVector<std::pair<mlir::Block *, bool>> worklist;
  worklist.push_back({&region.front(), entered});
  if (!region.hasOneBlock() && (mayHideCase || !targetedLabels.empty()))
    for (mlir::Block &block : llvm::drop_begin(region))
      if (llvm::any_of(block, [&](mlir::Operation &op) {
            return isTargetedLabel(&op) || mustVisitDead(op);
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
        if (isTargetedLabel(&op)) {
          live = true;
          continue;
        }
        if (!mustVisitDead(op)) {
          if (tailOnly && (op.getNumRegions() != 0 || mlir::isa<LabelOp>(op)))
            unsure = true;
          continue;
        }
      }

      if (op.getNumRegions() != 0) {
        Exits opExits = visitStructured(&op, live);
        exits.mergeEscaping(opExits);
        live = opExits.normal;
        continue;
      }

      if (!live)
        continue;

      if (isNoReturnCall(&op) || isAssumeFalse(&op)) {
        abnormalEdge = true;
        live = false;
      } else if (mlir::isa<ThrowOp>(op)) {
        fakeEdge = true;
        live = false;
      } else if (&op == terminator) {
        visitTerminator(&op, exits, liveSuccessors);
        live = false;
      } else if (tailOnly && mlir::isa<LabelOp>(op)) {
        unsure = true;
      }
    }

    // A block without a terminator falls through to the region's parent.
    if (live)
      exits.normal = true;

    for (mlir::Block *succ : liveSuccessors)
      worklist.push_back({succ, true});
  }
  if (tailOnly && visited.size() != llvm::range_size(region))
    unsure = true;
  return exits;
}

void FallOffAnalyzer::visitTerminator(
    mlir::Operation *op, Exits &exits,
    llvm::SmallVectorImpl<mlir::Block *> &liveSuccessors) {
  if (mlir::isa<YieldOp>(op)) {
    exits.normal = true;
  } else if (auto condOp = mlir::dyn_cast<ConditionOp>(op)) {
    std::optional<bool> cond = getConstantBool(condOp.getCondition());
    exits.condTrue |= cond.value_or(true);
    exits.condFalse |= !cond.value_or(false);
  } else if (mlir::isa<CoReturnOp>(op)) {
    exits.explicitReturn = true;
  } else if (auto retOp = mlir::dyn_cast<ReturnOp>(op)) {
    if (retOp.getImplicit())
      exits.implicitReturn = true;
    else
      exits.explicitReturn = true;
  } else if (mlir::isa<TrapOp, UnreachableOp>(op)) {
    // User traps and __builtin_unreachable are noreturn, like in the CFG.
    auto trapOp = mlir::dyn_cast<TrapOp>(op);
    bool isImplicit = trapOp ? trapOp.getImplicit()
                             : mlir::cast<UnreachableOp>(op).getImplicit();
    if (isImplicit)
      exits.implicitReturn = true;
    else
      abnormalEdge = true;
  } else if (mlir::isa<BreakOp>(op)) {
    exits.breaks = true;
  } else if (mlir::isa<ContinueOp>(op)) {
    exits.continues = true;
  } else if (auto brCond = mlir::dyn_cast<BrCondOp>(op)) {
    std::optional<bool> cond = getConstantBool(brCond.getCond());
    if (cond.value_or(true))
      liveSuccessors.push_back(brCond.getDestTrue());
    if (!cond.value_or(false))
      liveSuccessors.push_back(brCond.getDestFalse());
  } else if (auto gotoOp = mlir::dyn_cast<GotoOp>(op)) {
    reachedGotoTargets.insert(gotoOp.getLabelAttr());
  } else if (mlir::isa<IndirectGotoOp>(op)) {
    reachedIndirectGoto = true;
  } else {
    // Everything else with successors (cir.br, cir.switch.flat, ...)
    // reaches all of them.
    llvm::append_range(liveSuccessors, op->getSuccessors());
  }
}

Exits FallOffAnalyzer::visitStructured(mlir::Operation *op, bool entered) {
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
    std::optional<bool> cond = getConstantBool(ifOp.getCondition());
    Exits thenExits =
        visitRegion(ifOp.getThenRegion(), entered && cond.value_or(true));
    Exits elseExits =
        visitRegion(ifOp.getElseRegion(), entered && !cond.value_or(false));
    Exits exits;
    exits.normal = thenExits.normal || elseExits.normal;
    exits.mergeEscaping(thenExits);
    exits.mergeEscaping(elseExits);
    return exits;
  }

  if (auto tryOp = mlir::dyn_cast<TryOp>(op)) {
    // Like the CFG check, treat every handler as reachable when the try is.
    Exits exits = visitRegion(tryOp.getTryRegion(), entered);
    for (mlir::Region &handler : tryOp.getHandlerRegions()) {
      Exits handlerExits = visitRegion(handler, entered);
      exits.normal |= handlerExits.normal;
      exits.mergeEscaping(handlerExits);
    }
    return exits;
  }

  // Any region may run. Scopes and ternaries complete if a region does; ops
  // whose regions do not yield control complete whenever reached.
  Exits exits;
  bool yieldsControl = mlir::isa<ScopeOp, TernaryOp>(op);
  exits.normal = entered && !yieldsControl;
  for (mlir::Region &region : op->getRegions()) {
    Exits regionExits = visitRegion(region, entered);
    if (yieldsControl)
      exits.normal |= regionExits.normal;
    exits.mergeEscaping(regionExits);
  }
  return exits;
}

Exits FallOffAnalyzer::visitLoop(LoopOpInterface loop, bool entered) {
  bool isDoWhile = &loop.getEntry() == &loop.getBody();
  bool condEntered = entered && !isDoWhile;
  bool bodyEntered = entered && isDoWhile;
  mlir::Region *step = loop.maybeGetStep();

  // Visit the regions in execution order. Another round is only needed when
  // the loop was not entered from the top but through a goto into it.
  Exits condExits, bodyExits, stepExits;
  auto visitBodyAndStep = [&]() {
    bodyExits = visitRegion(loop.getBody(), bodyEntered);
    bool reachesLatch = bodyExits.normal || bodyExits.continues;
    if (step) {
      stepExits = visitRegion(*step, reachesLatch);
      reachesLatch = stepExits.normal;
    }
    return reachesLatch;
  };
  while (true) {
    if (isDoWhile) {
      condEntered |= visitBodyAndStep();
      condExits = visitRegion(loop.getCond(), condEntered);
      if (!condExits.condTrue || bodyEntered)
        break;
      bodyEntered = true;
    } else {
      condExits = visitRegion(loop.getCond(), condEntered);
      bodyEntered |= condExits.condTrue;
      if (!visitBodyAndStep() || condEntered)
        break;
      condEntered = true;
    }
  }

  Exits exits;
  exits.normal = condExits.condFalse || bodyExits.breaks;
  for (const Exits &regionExits : {condExits, bodyExits, stepExits}) {
    exits.explicitReturn |= regionExits.explicitReturn;
    exits.implicitReturn |= regionExits.implicitReturn;
  }
  return exits;
}

Exits FallOffAnalyzer::visitSwitch(SwitchOp op, bool entered) {
  // The switch body is entered through its cases only, so code ahead of the
  // first case is dead.
  switches.push_back({entered, /*hasDefault=*/false});
  Exits bodyExits = visitRegion(op.getBody(), /*entered=*/false);
  bool hasDefault = switches.pop_back_val().hasDefault;

  // As in the CFG check, a switch listing every enumerator has no implicit
  // default edge.
  Exits exits;
  exits.normal = bodyExits.normal || bodyExits.breaks ||
                 (entered && !hasDefault && !op.getAllEnumCasesCovered());
  exits.continues = bodyExits.continues;
  exits.explicitReturn = bodyExits.explicitReturn;
  exits.implicitReturn = bodyExits.implicitReturn;
  return exits;
}

Exits FallOffAnalyzer::visitCleanupScope(CleanupScopeOp op, bool entered) {
  Exits bodyExits = visitRegion(op.getBodyRegion(), entered);
  if (op.getCleanupKind() == CleanupKind::EH)
    return bodyExits;

  // A cleanup that cannot complete (e.g. a noreturn destructor) stops every
  // path leaving the body.
  Exits cleanupExits = visitRegion(op.getCleanupRegion(), bodyExits.any());
  if (!bodyExits.any() || cleanupExits.normal)
    return bodyExits;
  abnormalEdge = true;
  return Exits();
}

llvm::StringRef cir::stringifyFallOffKind(FallOffKind kind) {
  switch (kind) {
  case FallOffKind::Unknown:
    return "unknown";
  case FallOffKind::Never:
    return "never";
  case FallOffKind::NeverAndNoReturn:
    return "never-and-no-return";
  case FallOffKind::Maybe:
    return "maybe";
  case FallOffKind::Always:
    return "always";
  }
  llvm_unreachable("unknown FallOffKind");
}

static bool mayComplete(FuncOp func, mlir::Operation *op);

static bool isImplicitTerminator(mlir::Operation *op) {
  if (auto ret = mlir::dyn_cast<ReturnOp>(op))
    return ret.getImplicit();
  if (auto trap = mlir::dyn_cast<TrapOp>(op))
    return trap.getImplicit();
  if (auto unreachable = mlir::dyn_cast<UnreachableOp>(op))
    return unreachable.getImplicit();
  return false;
}

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
/// its body may reach its end, anything else analyzed on its own.
static bool mayComplete(FuncOp func, mlir::Operation *op) {
  mlir::Region *body = nullptr;
  if (auto scope = mlir::dyn_cast<ScopeOp>(op))
    body = &scope.getScopeRegion();
  else if (auto cleanupScope = mlir::dyn_cast<CleanupScopeOp>(op))
    body = &cleanupScope.getBodyRegion();
  if (!body)
    return FallOffAnalyzer(func, /*tailOnly=*/true).mayCompleteFromTop(op);
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

bool cir::mayFallOff(FuncOp func) {
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
    return isImplicitTerminator(terminator) &&
           mayReachPast(func, block, terminator->getPrevNode());
  });
}

FallOffKind cir::analyzeFallOff(FuncOp func) {
  // Declarations and aliases have no body to analyze.
  if (func.getBody().empty())
    return FallOffKind::Unknown;
  FallOffKind kind = FallOffAnalyzer(func).run();
#ifdef EXPENSIVE_CHECKS
  assert(((kind != FallOffKind::Maybe && kind != FallOffKind::Always) ||
          mayFallOff(func)) &&
         "mayFallOff rejected a function that falls off");
#endif
  return kind;
}

llvm::SmallVector<FallOffResult>
cir::analyzeFallOff(mlir::ModuleOp module,
                    llvm::function_ref<bool(FuncOp)> filter) {
  llvm::TimeTraceScope timeScope("CIRFallOffAnalysis");
  auto accept = [&](FuncOp func) {
    return !func.getBody().empty() && (!filter || filter(func));
  };
  llvm::SmallVector<FallOffResult> results;
  mlir::MLIRContext *context = module.getContext();
  if (!context->isMultithreadingEnabled()) {
    for (FuncOp func : module.getOps<FuncOp>())
      if (accept(func))
        results.push_back({func, analyzeFallOff(func)});
    return results;
  }

  // Filtering costs about as much as analyzing the few functions that pass,
  // so both run in parallel; each task writes only its own slot.
  llvm::SmallVector<FuncOp> funcs;
  for (FuncOp func : module.getOps<FuncOp>())
    funcs.push_back(func);
  llvm::SmallVector<std::optional<FallOffKind>> kinds(funcs.size());
  mlir::parallelFor(context, 0, funcs.size(), [&](size_t i) {
    if (accept(funcs[i]))
      kinds[i] = analyzeFallOff(funcs[i]);
  });
  for (auto [func, kind] : llvm::zip_equal(funcs, kinds))
    if (kind)
      results.push_back({func, *kind});
  return results;
}
