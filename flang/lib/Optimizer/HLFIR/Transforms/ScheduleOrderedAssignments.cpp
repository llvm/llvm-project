//===- ScheduleOrderedAssignments.cpp -- Ordered Assignment Scheduling ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScheduleOrderedAssignments.h"
#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-ordered-assignment"

//===----------------------------------------------------------------------===//
// Scheduling logging utilities for debug and test
//===----------------------------------------------------------------------===//

/// Log RAW or WAW conflict.
static void LLVM_ATTRIBUTE_UNUSED logConflict(llvm::raw_ostream &os,
                                              mlir::Value writtenOrReadVarA,
                                              mlir::Value writtenVarB);
/// Log when an expression evaluation must be saved.
static void LLVM_ATTRIBUTE_UNUSED logSaveEvaluation(llvm::raw_ostream &os,
                                                    unsigned runid,
                                                    mlir::Region &yieldRegion,
                                                    bool anyWrite);
/// Log when an assignment is scheduled.
static void LLVM_ATTRIBUTE_UNUSED logAssignmentEvaluation(
    llvm::raw_ostream &os, unsigned runid, hlfir::RegionAssignOp assign);
/// Log when starting to schedule an order assignment tree.
static void LLVM_ATTRIBUTE_UNUSED logStartScheduling(
    llvm::raw_ostream &os, hlfir::OrderedAssignmentTreeOpInterface root);
/// Log op if effect value is not known.
static void LLVM_ATTRIBUTE_UNUSED logIfUnkownEffectValue(
    llvm::raw_ostream &os, mlir::MemoryEffects::EffectInstance effect,
    mlir::Operation &op);

//===----------------------------------------------------------------------===//
// Scheduling Implementation
//===----------------------------------------------------------------------===//

namespace {
/// Structure that is in charge of building the schedule. For each
/// hlfir.region_assign inside an ordered assignment tree, it is walked through
/// the parent operations and their "leaf" regions (that contain expression
/// evaluations). The Scheduler analyze the memory effects of these regions
/// against the effect of the current assignment, and if any conflict is found,
/// it will create an action to save the value computed by the region before the
/// assignment evaluation.
class Scheduler {
public:
  Scheduler(bool tryFusingAssignments)
      : tryFusingAssignments{tryFusingAssignments} {}

  /// Start scheduling an assignment. Gather the write side effect from the
  /// assignment.
  void startSchedulingAssignment(hlfir::RegionAssignOp assign,
                                 bool leafRegionsMayOnlyRead);

  /// Start analysing a set of evaluation regions that can be evaluated in
  /// any order between themselves according to Fortran rules (like the controls
  /// of forall). The point of this is to avoid adding the side effects of
  /// independent evaluations to a run that would save only one of the control.
  void startIndependentEvaluationGroup() {
    assert(independentEvaluationEffects.empty() &&
           "previous group was not finished");
  };

  /// Analyze the memory effects of a region containing an expression
  /// evaluation. If any conflict is found with the current assignment, or if
  /// the expression has write effects (which is possible outside of forall),
  /// create an action in the schedule to save the value in the schedule before
  /// evaluating the current assignment. For expression with write effect,
  /// saving them ensures they are evaluated only once. A region whose value
  /// was saved in a previous run is considered to have no side effects with the
  /// current assignment: the saved value will be used.
  void saveEvaluationIfConflict(mlir::Region &yieldRegion,
                                bool leafRegionsMayOnlyRead,
                                bool yieldIsImplicitRead = true,
                                bool evaluationsMayConflict = false);

  /// Finish evaluating a group of independent regions. The current independent
  /// regions effects are added to the "parent" effect list since evaluating the
  /// next analyzed region would require evaluating the current independent
  /// regions.
  void finishIndependentEvaluationGroup() {
    parentEvaluationEffects.append(independentEvaluationEffects.begin(),
                                   independentEvaluationEffects.end());
    independentEvaluationEffects.clear();
  }

  /// After all the dependent evaluation regions have been analyzed, create the
  /// action to evaluate the assignment that was being analyzed.
  void finishSchedulingAssignment(hlfir::RegionAssignOp assign);

  /// Once all the assignments have been analyzed and scheduled, return the
  /// schedule. The scheduler object should not be used after this call.
  hlfir::Schedule moveSchedule() { return std::move(schedule); }

private:
  /// Save a conflicting region that is evaluating an expression that is
  /// controlling or masking the current assignment, or is evaluating the
  /// RHS/LHS.
  void
  saveEvaluation(mlir::Region &yieldRegion,
                 llvm::ArrayRef<mlir::MemoryEffects::EffectInstance> effects,
                 bool anyWrite);

  /// Can the current assignment be schedule with the previous run. This is
  /// only possible if the assignment and all of its dependencies have no side
  /// effects conflicting with the previous run.
  bool canFuseAssignmentWithPreviousRun();

  /// Memory effects of the assignments being lowered.
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> assignEffects;
  /// Memory effects of the evaluations implied by the assignments
  /// being lowered. They do not include the implicit writes
  /// to the LHS of the assignments.
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> assignEvaluateEffects;
  /// Memory effects of the unsaved evaluation region that are controlling or
  /// masking the current assignments.
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance>
      parentEvaluationEffects;
  /// Same as parentEvaluationEffects, but for the current "leaf group" being
  /// analyzed scheduled.
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance>
      independentEvaluationEffects;

  /// Were any region saved for the current assignment?
  bool savedAnyRegionForCurrentAssignment = false;

  // Schedule being built.
  hlfir::Schedule schedule;
  /// Leaf regions that have been saved so far.
  llvm::SmallSet<mlir::Region *, 16> savedRegions;
  /// Is schedule.back() a schedule that is only saving region with read
  /// effects?
  bool currentRunIsReadOnly = false;

  /// Option to tell if the scheduler should try fusing to assignments in the
  /// same loops.
  const bool tryFusingAssignments;
};
} // namespace

//===----------------------------------------------------------------------===//
// Scheduling Implementation : gathering memory effects of nodes.
//===----------------------------------------------------------------------===//

/// Is \p var the result of a ForallIndexOp?
/// Read effects to forall index can be ignored since forall
/// indices cannot be assigned to.
static bool isForallIndex(mlir::Value var) {
  return var &&
         mlir::isa_and_nonnull<hlfir::ForallIndexOp>(var.getDefiningOp());
}

/// Gather the memory effects of the operations contained in a region.
/// \p mayOnlyRead can be given to exclude some potential write effects that
/// cannot affect the current scheduling problem because it is known that the
/// regions are evaluating pure expressions from a Fortran point of view. It is
/// useful because low level IR in the region may contain operation that lacks
/// side effect interface, or that are writing temporary variables that may be
/// hard to identify as such (one would have to prove the write is "local" to
/// the region even when the alloca may be outside of the region).
static void gatherMemoryEffects(
    mlir::Region &region, bool mayOnlyRead,
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  /// This analysis is a simple walk of all the operations of the region that is
  /// evaluating and yielding a value. This is a lot simpler and safer than
  /// trying to walk back the SSA DAG from the yielded value. But if desired,
  /// this could be changed.
  for (mlir::Operation &op : region.getOps()) {
    if (op.hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>()) {
      for (mlir::Region &subRegion : op.getRegions())
        gatherMemoryEffects(subRegion, mayOnlyRead, effects);
      // In MLIR, RecursiveMemoryEffects can be combined with
      // MemoryEffectOpInterface to describe extra effects on top of the
      // effects of the nested operations.  However, the presence of
      // RecursiveMemoryEffects and the absence of MemoryEffectOpInterface
      // implies the operation has no other memory effects than the one of its
      // nested operations.
      if (!mlir::isa<mlir::MemoryEffectOpInterface>(op))
        continue;
    }
    mlir::MemoryEffectOpInterface interface =
        mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
    if (!interface) {
      LLVM_DEBUG(llvm::dbgs() << "unknown effect: " << op << "\n";);
      // There is no generic way to know what this operation is reading/writing
      // to. Assume the worst. No need to continue analyzing the code any
      // further.
      effects.emplace_back(mlir::MemoryEffects::Read::get());
      if (!mayOnlyRead)
        effects.emplace_back(mlir::MemoryEffects::Write::get());
      return;
    }
    // Collect read/write effects. Alloc/Free effects do not matter, they
    // are either local to the evaluation region and can be repeated, or, if
    // they are allocatable/pointer allocation/deallocation, they are conveyed
    // via the write that is updating the descriptor/allocatable (and there
    // cannot be any indirect allocatable/pointer allocation/deallocation if
    // mayOnlyRead is set). When mayOnlyRead is set, local write effects are
    // also ignored.
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance> opEffects;
    interface.getEffects(opEffects);
    for (auto &effect : opEffects)
      if (!isForallIndex(effect.getValue())) {
        if (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect())) {
          LLVM_DEBUG(logIfUnkownEffectValue(llvm::dbgs(), effect, op););
          effects.push_back(effect);
        } else if (!mayOnlyRead &&
                   mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
          LLVM_DEBUG(logIfUnkownEffectValue(llvm::dbgs(), effect, op););
          effects.push_back(effect);
        }
      }
  }
}

/// Return the entity yielded by a region, or a null value if the region
/// is not terminated by a yield.
static mlir::OpOperand *getYieldedEntity(mlir::Region &region) {
  if (region.empty() || region.back().empty())
    return nullptr;
  if (auto yield = mlir::dyn_cast<hlfir::YieldOp>(region.back().back()))
    return &yield.getEntityMutable();
  if (auto elementalAddr =
          mlir::dyn_cast<hlfir::ElementalAddrOp>(region.back().back()))
    return &elementalAddr.getYieldOp().getEntityMutable();
  return nullptr;
}

/// Gather the effect of an assignment. This is the implicit write to the LHS
/// of an assignment. This also includes the effects of the user defined
/// assignment, if any, but this does not include the effects of evaluating the
/// RHS and LHS, which occur before the assignment effects in Fortran.
static void gatherAssignEffects(
    hlfir::RegionAssignOp regionAssign,
    bool userDefAssignmentMayOnlyWriteToAssignedVariable,
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &assignEffects) {
  mlir::OpOperand *assignedVar = getYieldedEntity(regionAssign.getLhsRegion());
  assert(assignedVar && "lhs cannot be an empty region");
  assignEffects.emplace_back(mlir::MemoryEffects::Write::get(), assignedVar);

  if (!regionAssign.getUserDefinedAssignment().empty()) {
    // The write effect on the INTENT(OUT) LHS argument is already taken
    // into account above.
    // This side effects are "defensive" and could be improved.
    // On top of the passed RHS argument, user defined assignments (even when
    // pure) may also read host/used/common variable. Impure user defined
    // assignments may write to host/used/common variables not passed via
    // arguments. For now, simply assume the worst. Once fir.call side effects
    // analysis is improved, it would best to let the call side effects be used
    // directly.
    if (userDefAssignmentMayOnlyWriteToAssignedVariable)
      assignEffects.emplace_back(mlir::MemoryEffects::Read::get());
    else
      assignEffects.emplace_back(mlir::MemoryEffects::Write::get());
  }
}

/// Gather the effects of evaluations implied by the given assignment.
/// These are the effects of operations from LHS and RHS.
static void gatherAssignEvaluationEffects(
    hlfir::RegionAssignOp regionAssign,
    bool userDefAssignmentMayOnlyWriteToAssignedVariable,
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &assignEffects) {
  gatherMemoryEffects(regionAssign.getLhsRegion(),
                      userDefAssignmentMayOnlyWriteToAssignedVariable,
                      assignEffects);
  gatherMemoryEffects(regionAssign.getRhsRegion(),
                      userDefAssignmentMayOnlyWriteToAssignedVariable,
                      assignEffects);
}

//===----------------------------------------------------------------------===//
// Scheduling Implementation : finding conflicting memory effects.
//===----------------------------------------------------------------------===//

/// Follow addressing and declare like operation to the storage source.
/// This allows using FIR alias analysis that otherwise does not know
/// about those operations. This is correct, but ignoring the designate
/// and declare info may yield false positive regarding aliasing (e.g,
/// if it could be proved that the variable are different sub-part of
/// an array).
static mlir::Value getStorageSource(mlir::Value var) {
  // TODO: define some kind of View interface for Fortran in FIR,
  // and use it in the FIR alias analysis.
  mlir::Value source = var;
  while (auto *op = source.getDefiningOp()) {
    if (auto designate = mlir::dyn_cast<hlfir::DesignateOp>(op)) {
      source = designate.getMemref();
    } else if (auto declare = mlir::dyn_cast<hlfir::DeclareOp>(op)) {
      source = declare.getMemref();
    } else {
      break;
    }
  }
  return source;
}

/// Could there be any read or write in effectsA on a variable written to in
/// effectsB?
static bool
anyRAWorWAW(llvm::ArrayRef<mlir::MemoryEffects::EffectInstance> effectsA,
            llvm::ArrayRef<mlir::MemoryEffects::EffectInstance> effectsB,
            fir::AliasAnalysis &aliasAnalysis) {
  for (const auto &effectB : effectsB)
    if (mlir::isa<mlir::MemoryEffects::Write>(effectB.getEffect())) {
      mlir::Value writtenVarB = effectB.getValue();
      if (writtenVarB)
        writtenVarB = getStorageSource(writtenVarB);
      for (const auto &effectA : effectsA)
        if (mlir::isa<mlir::MemoryEffects::Write, mlir::MemoryEffects::Read>(
                effectA.getEffect())) {
          mlir::Value writtenOrReadVarA = effectA.getValue();
          if (!writtenVarB || !writtenOrReadVarA) {
            LLVM_DEBUG(
                logConflict(llvm::dbgs(), writtenOrReadVarA, writtenVarB););
            return true; // unknown conflict.
          }
          writtenOrReadVarA = getStorageSource(writtenOrReadVarA);
          if (!aliasAnalysis.alias(writtenOrReadVarA, writtenVarB).isNo()) {
            LLVM_DEBUG(
                logConflict(llvm::dbgs(), writtenOrReadVarA, writtenVarB););
            return true;
          }
        }
    }
  return false;
}

/// Could there be any read or write in effectsA on a variable written to in
/// effectsB, or any read in effectsB on a variable written to in effectsA?
static bool
conflict(llvm::ArrayRef<mlir::MemoryEffects::EffectInstance> effectsA,
         llvm::ArrayRef<mlir::MemoryEffects::EffectInstance> effectsB) {
  fir::AliasAnalysis aliasAnalysis;
  // (RAW || WAW) || (WAR || WAW).
  return anyRAWorWAW(effectsA, effectsB, aliasAnalysis) ||
         anyRAWorWAW(effectsB, effectsA, aliasAnalysis);
}

/// Could there be any write effects in "effects" affecting memory storages
/// that are not local to the current region.
static bool
anyNonLocalWrite(llvm::ArrayRef<mlir::MemoryEffects::EffectInstance> effects,
                 mlir::Region &region) {
  return llvm::any_of(
      effects, [&region](const mlir::MemoryEffects::EffectInstance &effect) {
        if (mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
          if (mlir::Value v = effect.getValue()) {
            v = getStorageSource(v);
            if (v.getDefiningOp<fir::AllocaOp>() ||
                v.getDefiningOp<fir::AllocMemOp>())
              return !region.isAncestor(v.getParentRegion());
          }
          return true;
        }
        return false;
      });
}

//===----------------------------------------------------------------------===//
// Scheduling Implementation : Scheduler class implementation
//===----------------------------------------------------------------------===//

void Scheduler::startSchedulingAssignment(hlfir::RegionAssignOp assign,
                                          bool leafRegionsMayOnlyRead) {
  gatherAssignEffects(assign, leafRegionsMayOnlyRead, assignEffects);
  // Unconditionally collect effects of the evaluations of LHS and RHS
  // in case they need to be analyzed for any parent that might be
  // affected by conflicts of these evaluations.
  // This collection migth be skipped, if there are no such parents,
  // but for the time being we run it always.
  gatherAssignEvaluationEffects(assign, leafRegionsMayOnlyRead,
                                assignEvaluateEffects);
}

void Scheduler::saveEvaluationIfConflict(mlir::Region &yieldRegion,
                                         bool leafRegionsMayOnlyRead,
                                         bool yieldIsImplicitRead,
                                         bool evaluationsMayConflict) {
  // If the region evaluation was previously executed and saved, the saved
  // value will be used when evaluating the current assignment and this has
  // no effects in the current assignment evaluation.
  if (savedRegions.contains(&yieldRegion))
    return;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  gatherMemoryEffects(yieldRegion, leafRegionsMayOnlyRead, effects);
  // Yield has no effect as such, but in the context of order assignments.
  // The order assignments will usually read the yielded entity (except for
  // the yielded assignments LHS that is only read if this is an assignment
  // with a finalizer, or a user defined assignment where the LHS is
  // intent(inout)).
  if (yieldIsImplicitRead) {
    mlir::OpOperand *entity = getYieldedEntity(yieldRegion);
    if (entity && hlfir::isFortranVariableType(entity->get().getType()))
      effects.emplace_back(mlir::MemoryEffects::Read::get(), entity);
  }
  if (!leafRegionsMayOnlyRead && anyNonLocalWrite(effects, yieldRegion)) {
    // Region with write effect must be executed only once (unless all writes
    // affect storages allocated inside the region): save it the first time it
    // is encountered.
    LLVM_DEBUG(llvm::dbgs()
                   << "saving eval because write effect prevents re-evaluation"
                   << "\n";);
    saveEvaluation(yieldRegion, effects, /*anyWrite=*/true);
  } else if (conflict(effects, assignEffects)) {
    // Region that conflicts with the current assignments must be fully
    // evaluated and saved before doing the assignment (Note that it may
    // have already have been evaluated without saving it before, but this
    // implies that it never conflicted with a prior assignment, so its value
    // should be the same.)
    saveEvaluation(yieldRegion, effects, /*anyWrite=*/false);
  } else if (evaluationsMayConflict &&
             conflict(effects, assignEvaluateEffects)) {
    // If evaluations of the assignment may conflict with the yield
    // evaluations, we have to save yield evaluation.
    // For example, a WHERE mask might be written by the masked assignment
    // evaluations, and it has to be saved in this case:
    //   where (mask) r = f() ! function f modifies mask
    saveEvaluation(yieldRegion, effects,
                   anyNonLocalWrite(effects, yieldRegion));
  } else {
    // Can be executed while doing the assignment.
    independentEvaluationEffects.append(effects.begin(), effects.end());
  }
}

void Scheduler::saveEvaluation(
    mlir::Region &yieldRegion,
    llvm::ArrayRef<mlir::MemoryEffects::EffectInstance> effects,
    bool anyWrite) {
  savedAnyRegionForCurrentAssignment = true;
  if (anyWrite) {
    // Create a new run just for regions with side effect. Further analysis
    // could try to prove the effects do not conflict with the previous
    // schedule.
    schedule.emplace_back(hlfir::Run{});
    currentRunIsReadOnly = false;
  } else if (!currentRunIsReadOnly) {
    // For now, do not try to fuse an evaluation with a previous
    // run that contains any write effects. One could try to prove
    // that "effects" do not conflict with the current run assignments.
    schedule.emplace_back(hlfir::Run{});
    currentRunIsReadOnly = true;
  }
  // Otherwise, save the yielded entity in the current run, that already
  // saving other read only entities.
  schedule.back().actions.emplace_back(hlfir::SaveEntity{&yieldRegion});
  // The run to save the yielded entity will need to evaluate all the unsaved
  // parent control or masks. Note that these effects may already be in the
  // current run memoryEffects, but it is just easier always add them, even if
  // this may add them again.
  schedule.back().memoryEffects.append(parentEvaluationEffects.begin(),
                                       parentEvaluationEffects.end());
  schedule.back().memoryEffects.append(effects.begin(), effects.end());
  savedRegions.insert(&yieldRegion);
  LLVM_DEBUG(
      logSaveEvaluation(llvm::dbgs(), schedule.size(), yieldRegion, anyWrite););
}

bool Scheduler::canFuseAssignmentWithPreviousRun() {
  // If a region was saved for the current assignment, the previous
  // run is already known to conflict. Skip the analysis.
  if (savedAnyRegionForCurrentAssignment || schedule.empty())
    return false;
  auto &previousRunEffects = schedule.back().memoryEffects;
  return !conflict(previousRunEffects, assignEffects) &&
         !conflict(previousRunEffects, parentEvaluationEffects) &&
         !conflict(previousRunEffects, independentEvaluationEffects);
}

void Scheduler::finishSchedulingAssignment(hlfir::RegionAssignOp assign) {
  // For now, always schedule each assignment in its own run. They could
  // be done as part of previous assignment runs if it is proven they have
  // no conflicting effects.
  currentRunIsReadOnly = false;
  if (!tryFusingAssignments || !canFuseAssignmentWithPreviousRun())
    schedule.emplace_back(hlfir::Run{});
  schedule.back().actions.emplace_back(assign);
  // TODO: when fusing, it would probably be best to filter the
  // parentEvaluationEffects that already in the previous run effects (since
  // assignments may share the same parents), otherwise, this can make the
  // conflict() calls more and more expensive.
  schedule.back().memoryEffects.append(parentEvaluationEffects.begin(),
                                       parentEvaluationEffects.end());
  schedule.back().memoryEffects.append(assignEffects.begin(),
                                       assignEffects.end());
  assignEffects.clear();
  assignEvaluateEffects.clear();
  parentEvaluationEffects.clear();
  independentEvaluationEffects.clear();
  savedAnyRegionForCurrentAssignment = false;
  LLVM_DEBUG(logAssignmentEvaluation(llvm::dbgs(), schedule.size(), assign));
}

//===----------------------------------------------------------------------===//
// Scheduling Implementation : driving the Scheduler in the assignment tree.
//===----------------------------------------------------------------------===//

/// Gather the hlfir.region_assign nested directly and indirectly inside root in
/// execution order.
static void
gatherAssignments(hlfir::OrderedAssignmentTreeOpInterface root,
                  llvm::SmallVector<hlfir::RegionAssignOp> &assignments) {
  llvm::SmallVector<mlir::Operation *> nodeStack{root.getOperation()};
  while (!nodeStack.empty()) {
    mlir::Operation *node = nodeStack.pop_back_val();
    if (auto regionAssign = mlir::dyn_cast<hlfir::RegionAssignOp>(node)) {
      assignments.push_back(regionAssign);
      continue;
    }
    auto nodeIface =
        mlir::dyn_cast<hlfir::OrderedAssignmentTreeOpInterface>(node);
    if (nodeIface)
      if (mlir::Block *block = nodeIface.getSubTreeBlock())
        for (mlir::Operation &op : llvm::reverse(block->getOperations()))
          nodeStack.push_back(&op);
  }
}

/// Gather the parents of (not included) \p node in reverse execution order.
static void gatherParents(
    hlfir::OrderedAssignmentTreeOpInterface node,
    llvm::SmallVectorImpl<hlfir::OrderedAssignmentTreeOpInterface> &parents) {
  while (node) {
    auto parent =
        mlir::dyn_cast_or_null<hlfir::OrderedAssignmentTreeOpInterface>(
            node->getParentOp());
    if (parent && parent.getSubTreeRegion() == node->getParentRegion()) {
      parents.push_back(parent);
      node = parent;
    } else {
      break;
    }
  }
}

// Build the list of the parent nodes for this assignment. The list is built
// from the closest parent until the ordered assignment tree root (this is the
// revere of their execution order).
static void gatherAssignmentParents(
    hlfir::RegionAssignOp assign,
    llvm::SmallVectorImpl<hlfir::OrderedAssignmentTreeOpInterface> &parents) {
  gatherParents(mlir::cast<hlfir::OrderedAssignmentTreeOpInterface>(
                    assign.getOperation()),
                parents);
}

hlfir::Schedule
hlfir::buildEvaluationSchedule(hlfir::OrderedAssignmentTreeOpInterface root,
                               bool tryFusingAssignments) {
  LLVM_DEBUG(logStartScheduling(llvm::dbgs(), root););
  // The expressions inside an hlfir.forall must be pure (with the Fortran
  // definition of pure). This is not a commitment that there are no operation
  // with write effect in the regions: entities local to the region may still
  // be written to (e.g., a temporary accumulator implementing SUM). This is
  // a commitment that no write effect will affect the scheduling problem, and
  // that all write effect caught by MLIR analysis can be ignored for the
  // current problem.
  const bool leafRegionsMayOnlyRead =
      mlir::isa<hlfir::ForallOp>(root.getOperation());

  // Loop through the assignments and schedule them.
  Scheduler scheduler(tryFusingAssignments);
  llvm::SmallVector<hlfir::RegionAssignOp> assignments;
  gatherAssignments(root, assignments);
  for (hlfir::RegionAssignOp assign : assignments) {
    scheduler.startSchedulingAssignment(assign, leafRegionsMayOnlyRead);
    // Go through the list of parents (not including the current
    // hlfir.region_assign) in Fortran execution order so that any parent leaf
    // region that must be saved is saved in order.
    llvm::SmallVector<hlfir::OrderedAssignmentTreeOpInterface> parents;
    gatherAssignmentParents(assign, parents);
    for (hlfir::OrderedAssignmentTreeOpInterface parent :
         llvm::reverse(parents)) {
      scheduler.startIndependentEvaluationGroup();
      llvm::SmallVector<mlir::Region *, 4> yieldRegions;
      parent.getLeafRegions(yieldRegions);
      // TODO: is this really limited to WHERE/ELSEWHERE?
      bool evaluationsMayConflict = mlir::isa<hlfir::WhereOp>(parent) ||
                                    mlir::isa<hlfir::ElseWhereOp>(parent);
      for (mlir::Region *yieldRegion : yieldRegions)
        scheduler.saveEvaluationIfConflict(*yieldRegion, leafRegionsMayOnlyRead,
                                           /*yieldIsImplicitRead=*/true,
                                           evaluationsMayConflict);
      scheduler.finishIndependentEvaluationGroup();
    }
    // Look for conflicts between the RHS/LHS evaluation and the assignments.
    // The LHS yield has no implicit read effect on the produced variable (the
    // variable is not read before the assignment).
    scheduler.startIndependentEvaluationGroup();
    scheduler.saveEvaluationIfConflict(assign.getRhsRegion(),
                                       leafRegionsMayOnlyRead);
    // There is no point to save the LHS outside of Forall and assignment to a
    // vector subscripted LHS because the LHS is already fully evaluated and
    // saved in the resulting SSA address value (that may be a descriptor or
    // descriptor address).
    if (mlir::isa<hlfir::ForallOp>(root.getOperation()) ||
        mlir::isa<hlfir::ElementalAddrOp>(assign.getLhsRegion().back().back()))
      scheduler.saveEvaluationIfConflict(assign.getLhsRegion(),
                                         leafRegionsMayOnlyRead,
                                         /*yieldIsImplicitRead=*/false);
    scheduler.finishIndependentEvaluationGroup();
    scheduler.finishSchedulingAssignment(assign);
  }
  return scheduler.moveSchedule();
}

mlir::Value hlfir::SaveEntity::getSavedValue() {
  mlir::OpOperand *saved = getYieldedEntity(*yieldRegion);
  assert(saved && "SaveEntity must contain region terminated by YieldOp");
  return saved->get();
}

//===----------------------------------------------------------------------===//
// Debug and test logging implementation
//===----------------------------------------------------------------------===//

static llvm::raw_ostream &printRegionId(llvm::raw_ostream &os,
                                        mlir::Region &yieldRegion) {
  mlir::Operation *parent = yieldRegion.getParentOp();
  if (auto forall = mlir::dyn_cast<hlfir::ForallOp>(parent)) {
    if (&forall.getLbRegion() == &yieldRegion)
      os << "lb";
    else if (&forall.getUbRegion() == &yieldRegion)
      os << "ub";
    else if (&forall.getStepRegion() == &yieldRegion)
      os << "step";
  } else if (auto assign = mlir::dyn_cast<hlfir::ForallMaskOp>(parent)) {
    if (&assign.getMaskRegion() == &yieldRegion)
      os << "mask";
  } else if (auto assign = mlir::dyn_cast<hlfir::RegionAssignOp>(parent)) {
    if (&assign.getRhsRegion() == &yieldRegion)
      os << "rhs";
    else if (&assign.getLhsRegion() == &yieldRegion)
      os << "lhs";
  } else if (auto where = mlir::dyn_cast<hlfir::WhereOp>(parent)) {
    if (&where.getMaskRegion() == &yieldRegion)
      os << "mask";
  } else if (auto elseWhereOp = mlir::dyn_cast<hlfir::ElseWhereOp>(parent)) {
    if (&elseWhereOp.getMaskRegion() == &yieldRegion)
      os << "mask";
  } else {
    os << "unknown";
  }
  return os;
}

static llvm::raw_ostream &
printNodeIndexInBody(llvm::raw_ostream &os,
                     hlfir::OrderedAssignmentTreeOpInterface node,
                     hlfir::OrderedAssignmentTreeOpInterface parent) {
  if (!parent || !parent.getSubTreeRegion())
    return os;
  mlir::Operation *nodeOp = node.getOperation();
  unsigned index = 1;
  for (mlir::Operation &op : parent.getSubTreeRegion()->getOps())
    if (nodeOp == &op) {
      return os << index;
    } else if (nodeOp->getName() == op.getName()) {
      ++index;
    }
  return os;
}

static llvm::raw_ostream &printNodePath(llvm::raw_ostream &os,
                                        mlir::Operation *op) {
  auto node =
      mlir::dyn_cast_or_null<hlfir::OrderedAssignmentTreeOpInterface>(op);
  if (!node) {
    os << "unknown node";
    return os;
  }
  llvm::SmallVector<hlfir::OrderedAssignmentTreeOpInterface> parents;
  gatherParents(node, parents);
  hlfir::OrderedAssignmentTreeOpInterface previousParent;
  for (auto parent : llvm::reverse(parents)) {
    os << parent->getName().stripDialect();
    printNodeIndexInBody(os, parent, previousParent) << "/";
    previousParent = parent;
  }
  os << node->getName().stripDialect();
  return printNodeIndexInBody(os, node, previousParent);
}

static llvm::raw_ostream &printRegionPath(llvm::raw_ostream &os,
                                          mlir::Region &yieldRegion) {
  printNodePath(os, yieldRegion.getParentOp()) << "/";
  return printRegionId(os, yieldRegion);
}

static void LLVM_ATTRIBUTE_UNUSED logSaveEvaluation(llvm::raw_ostream &os,
                                                    unsigned runid,
                                                    mlir::Region &yieldRegion,
                                                    bool anyWrite) {
  os << "run " << runid << " save  " << (anyWrite ? "(w)" : "  ") << ": ";
  printRegionPath(os, yieldRegion) << "\n";
}

static void LLVM_ATTRIBUTE_UNUSED logAssignmentEvaluation(
    llvm::raw_ostream &os, unsigned runid, hlfir::RegionAssignOp assign) {
  os << "run " << runid << " evaluate: ";
  printNodePath(os, assign.getOperation()) << "\n";
}

static void LLVM_ATTRIBUTE_UNUSED logConflict(llvm::raw_ostream &os,
                                              mlir::Value writtenOrReadVarA,
                                              mlir::Value writtenVarB) {
  auto printIfValue = [&](mlir::Value var) -> llvm::raw_ostream & {
    if (!var)
      return os << "<unknown>";
    return os << var;
  };
  os << "conflict: R/W: ";
  printIfValue(writtenOrReadVarA) << " W:";
  printIfValue(writtenVarB) << "\n";
}

static void LLVM_ATTRIBUTE_UNUSED logStartScheduling(
    llvm::raw_ostream &os, hlfir::OrderedAssignmentTreeOpInterface root) {
  os << "------------ scheduling ";
  printNodePath(os, root.getOperation());
  if (auto funcOp = root->getParentOfType<mlir::func::FuncOp>())
    os << " in " << funcOp.getSymName() << " ";
  os << "------------\n";
}

static void LLVM_ATTRIBUTE_UNUSED logIfUnkownEffectValue(
    llvm::raw_ostream &os, mlir::MemoryEffects::EffectInstance effect,
    mlir::Operation &op) {
  if (effect.getValue() != nullptr)
    return;
  os << "unknown effected value (";
  os << (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()) ? "R" : "W");
  os << "): " << op << "\n";
}
