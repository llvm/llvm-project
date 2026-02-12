//===- ScheduleOrderedAssignments.cpp -- Ordered Assignment Scheduling ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScheduleOrderedAssignments.h"
#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Analysis/ArraySectionAnalyzer.h"
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
[[maybe_unused]] static void logConflict(llvm::raw_ostream &os,
                                         mlir::Value writtenOrReadVarA,
                                         mlir::Value writtenVarB,
                                         bool isAligned = false);
/// Log when a region must be retroactively saved.
[[maybe_unused]] static void
logRetroactiveSave(llvm::raw_ostream &os, mlir::Region &yieldRegion,
                   hlfir::Run &modifyingRun,
                   hlfir::RegionAssignOp currentAssign);
/// Log when an expression evaluation must be saved.
[[maybe_unused]] static void logSaveEvaluation(llvm::raw_ostream &os,
                                               unsigned runid,
                                               mlir::Region &yieldRegion,
                                               bool anyWrite);
/// Log when an assignment is scheduled.
[[maybe_unused]] static void
logAssignmentEvaluation(llvm::raw_ostream &os, unsigned runid,
                        hlfir::RegionAssignOp assign);
/// Log when starting to schedule an order assignment tree.
[[maybe_unused]] static void
logStartScheduling(llvm::raw_ostream &os,
                   hlfir::OrderedAssignmentTreeOpInterface root);
/// Log op if effect value is not known.
[[maybe_unused]] static void
logIfUnknownEffectValue(llvm::raw_ostream &os,
                        mlir::MemoryEffects::EffectInstance effect,
                        mlir::Operation &op);

//===----------------------------------------------------------------------===//
// Scheduling Implementation
//===----------------------------------------------------------------------===//

/// Is the apply using all the elemental indices in order?
static bool isInOrderApply(hlfir::ApplyOp apply,
                           hlfir::ElementalOpInterface elemental) {
  mlir::Region::BlockArgListType elementalIndices = elemental.getIndices();
  if (elementalIndices.size() != apply.getIndices().size())
    return false;
  for (auto [elementalIdx, applyIdx] :
       llvm::zip(elementalIndices, apply.getIndices()))
    if (elementalIdx != applyIdx)
      return false;
  return true;
}

hlfir::ElementalTree
hlfir::ElementalTree::buildElementalTree(mlir::Operation &regionTerminator) {
  ElementalTree tree;
  if (auto elementalAddr =
          mlir::dyn_cast<hlfir::ElementalOpInterface>(regionTerminator)) {
    // Vector subscripted designator (hlfir.elemental_addr terminator).
    tree.gatherElementalTree(elementalAddr, /*isAppliedInOrder=*/true);
    return tree;
  }
  // Try if elemental expression.
  if (auto yield = mlir::dyn_cast<hlfir::YieldOp>(regionTerminator)) {
    mlir::Value entity = yield.getEntity();
    if (auto maybeElemental =
            mlir::dyn_cast_or_null<hlfir::ElementalOpInterface>(
                entity.getDefiningOp()))
      tree.gatherElementalTree(maybeElemental, /*isAppliedInOrder=*/true);
  }
  return tree;
}

// Check if op is an ElementalOpInterface that is part of this elemental tree.
bool hlfir::ElementalTree::contains(mlir::Operation *op) const {
  for (auto &p : tree)
    if (p.first == op)
      return true;
  return false;
}

std::optional<bool> hlfir::ElementalTree::isOrdered(mlir::Operation *op) const {
  for (auto &p : tree)
    if (p.first == op)
      return p.second;
  return std::nullopt;
}

void hlfir::ElementalTree::gatherElementalTree(
    hlfir::ElementalOpInterface elemental, bool isAppliedInOrder) {
  if (!elemental)
    return;
  // Only inline an applied elemental that must be executed in order if the
  // applying indices are in order. An hlfir::Elemental may have been created
  // for a transformational like transpose, and Fortran 2018 standard
  // section 10.2.3.2, point 10 imply that impure elemental sub-expression
  // evaluations should not be masked if they are the arguments of
  // transformational expressions.
  if (!isAppliedInOrder && elemental.isOrdered())
    return;

  insert(elemental, isAppliedInOrder);
  for (mlir::Operation &op : elemental.getElementalRegion().getOps())
    if (auto apply = mlir::dyn_cast<hlfir::ApplyOp>(op)) {
      bool isUnorderedApply =
          !isAppliedInOrder || !isInOrderApply(apply, elemental);
      auto maybeElemental = mlir::dyn_cast_or_null<hlfir::ElementalOpInterface>(
          apply.getExpr().getDefiningOp());
      gatherElementalTree(maybeElemental, !isUnorderedApply);
    }
}

void hlfir::ElementalTree::insert(hlfir::ElementalOpInterface elementalOp,
                                  bool isAppliedInOrder) {
  tree.push_back({elementalOp.getOperation(), isAppliedInOrder});
}

static bool isInOrderDesignate(hlfir::DesignateOp designate,
                               hlfir::ElementalTree *tree) {
  if (!tree)
    return false;
  if (auto elemental =
          designate->getParentOfType<hlfir::ElementalOpInterface>())
    if (tree->isOrdered(elemental.getOperation()))
      return fir::ArraySectionAnalyzer::isDesignatingArrayInOrder(designate,
                                                                  elemental);
  return false;
}

hlfir::DetailedEffectInstance::DetailedEffectInstance(
    mlir::MemoryEffects::Effect *effect, mlir::OpOperand *value,
    mlir::Value orderedElementalEffectOn)
    : effectInstance(effect, value),
      orderedElementalEffectOn(orderedElementalEffectOn) {}

hlfir::DetailedEffectInstance::DetailedEffectInstance(
    mlir::MemoryEffects::EffectInstance effectInst,
    mlir::Value orderedElementalEffectOn)
    : effectInstance(effectInst),
      orderedElementalEffectOn(orderedElementalEffectOn) {}

hlfir::DetailedEffectInstance
hlfir::DetailedEffectInstance::getArrayReadEffect(mlir::OpOperand *array) {
  return DetailedEffectInstance(mlir::MemoryEffects::Read::get(), array,
                                array->get());
}

hlfir::DetailedEffectInstance
hlfir::DetailedEffectInstance::getArrayWriteEffect(mlir::OpOperand *array) {
  return DetailedEffectInstance(mlir::MemoryEffects::Write::get(), array,
                                array->get());
}

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
  void finishSchedulingAssignment(hlfir::RegionAssignOp assign,
                                  bool leafRegionsMayOnlyRead);

  /// Once all the assignments have been analyzed and scheduled, return the
  /// schedule. The scheduler object should not be used after this call.
  hlfir::Schedule moveSchedule() { return std::move(schedule); }

private:
  struct EvaluationState {
    bool saved = false;
    std::optional<hlfir::Schedule::iterator> modifiedInRun;
  };

  /// Save a conflicting region that is evaluating an expression that is
  /// controlling or masking the current assignment, or is evaluating the
  /// RHS/LHS.
  void saveEvaluation(mlir::Region &yieldRegion,
                      llvm::ArrayRef<hlfir::DetailedEffectInstance> effects,
                      bool anyWrite);

  /// Can the current assignment be schedule with the previous run. This is
  /// only possible if the assignment and all of its dependencies have no side
  /// effects conflicting with the previous run.
  bool canFuseAssignmentWithPreviousRun();

  /// Memory effects of the assignments being lowered.
  llvm::SmallVector<hlfir::DetailedEffectInstance> assignEffects;
  /// Memory effects of the evaluations implied by the assignments
  /// being lowered. They do not include the implicit writes
  /// to the LHS of the assignments.
  llvm::SmallVector<hlfir::DetailedEffectInstance> assignEvaluateEffects;
  /// Memory effects of the unsaved evaluation region that are controlling or
  /// masking the current assignments.
  llvm::SmallVector<hlfir::DetailedEffectInstance> parentEvaluationEffects;
  /// Same as parentEvaluationEffects, but for the current "leaf group" being
  /// analyzed scheduled.
  llvm::SmallVector<hlfir::DetailedEffectInstance> independentEvaluationEffects;

  /// Were any region saved for the current assignment?
  bool savedAnyRegionForCurrentAssignment = false;

  // Schedule being built.
  hlfir::Schedule schedule;
  /// Leaf regions that have been saved so far.
  llvm::DenseMap<mlir::Region *, EvaluationState> regionStates;
  /// Regions that have an aligned conflict with the current assignment.
  llvm::SmallVector<mlir::Region *> pendingAlignedRegions;

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
static void gatherMemoryEffectsImpl(
    mlir::Region &region, bool mayOnlyRead,
    llvm::SmallVectorImpl<hlfir::DetailedEffectInstance> &effects,
    hlfir::ElementalTree *tree = nullptr) {
  /// This analysis is a simple walk of all the operations of the region that is
  /// evaluating and yielding a value. This is a lot simpler and safer than
  /// trying to walk back the SSA DAG from the yielded value. But if desired,
  /// this could be changed.
  for (mlir::Operation &op : region.getOps()) {
    if (op.hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>()) {
      for (mlir::Region &subRegion : op.getRegions())
        gatherMemoryEffectsImpl(subRegion, mayOnlyRead, effects, tree);
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
        mlir::Value array;
        if (effect.getValue())
          if (auto designate =
                  effect.getValue().getDefiningOp<hlfir::DesignateOp>())
            if (isInOrderDesignate(designate, tree))
              array = designate.getMemref();

        if (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect())) {
          LLVM_DEBUG(logIfUnknownEffectValue(llvm::dbgs(), effect, op););
          effects.emplace_back(effect, array);
        } else if (!mayOnlyRead &&
                   mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
          LLVM_DEBUG(logIfUnknownEffectValue(llvm::dbgs(), effect, op););
          effects.emplace_back(effect, array);
        }
      }
  }
}
static void gatherMemoryEffects(
    mlir::Region &region, bool mayOnlyRead,
    llvm::SmallVectorImpl<hlfir::DetailedEffectInstance> &effects) {
  if (!region.getParentOfType<hlfir::ForallOp>()) {
    // TODO: leverage array access analysis for FORALL.
    // While FORALL assignments can be array assignments, the iteration space
    // is also driven by the FORALL indices, so the way ArraySectionAnalyzer
    // results are used is not adequate for it.
    // For instance "disjoint" array access cannot be ignored in:
    // "forall (i=1:10) x(i+1,:) = x(i,:)".
    // While identical access can probably also be accepted, this would deserve
    // more thinking, it would probably make sense to also deal with "aligned
    // scalar" access for them like in "forall (i=1:10) x(i) = x(i) + 1".  For
    // now this feature is disabled for inside FORALL.
    hlfir::ElementalTree tree =
        hlfir::ElementalTree::buildElementalTree(region.back().back());
    gatherMemoryEffectsImpl(region, mayOnlyRead, effects, &tree);
    return;
  }
  gatherMemoryEffectsImpl(region, mayOnlyRead, effects, /*tree=*/nullptr);
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
    llvm::SmallVectorImpl<hlfir::DetailedEffectInstance> &assignEffects) {
  mlir::OpOperand *assignedVar = getYieldedEntity(regionAssign.getLhsRegion());
  assert(assignedVar && "lhs cannot be an empty region");
  if (regionAssign->getParentOfType<hlfir::ForallOp>())
    assignEffects.emplace_back(mlir::MemoryEffects::Write::get(), assignedVar);
  else
    assignEffects.emplace_back(
        hlfir::DetailedEffectInstance::getArrayWriteEffect(assignedVar));

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
    llvm::SmallVectorImpl<hlfir::DetailedEffectInstance> &assignEffects) {
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

namespace {

/// Class to represent conflicts between several accesses (effects) to a memory
/// location (read after write, write after write).
struct ConflictKind {
  enum Kind {
    // None: The effects are not affecting the same memory location, or they are
    // all reads.
    None,
    // Aligned: There are both read and write effects affecting the same memory
    // location, but it is known that these effects are all accessing the memory
    // location element by element in array order. This means the conflict does
    // not introduce loop-carried dependencies.
    Aligned,
    // Any: There may be both read and write effects affecting the same memory
    // in any way.
    Any
  };
  Kind kind;

  ConflictKind(Kind k) : kind(k) {}

  static ConflictKind none() { return ConflictKind(None); }
  static ConflictKind aligned() { return ConflictKind(Aligned); }
  static ConflictKind any() { return ConflictKind(Any); }

  bool isNone() const { return kind == None; }
  bool isAligned() const { return kind == Aligned; }
  bool isAny() const { return kind == Any; }

  // Merge conflicts:
  // none || none -> none
  // aligned || <not any> -> aligned
  // any || _ -> any
  ConflictKind operator||(const ConflictKind &other) const {
    if (kind == Any || other.kind == Any)
      return any();
    if (kind == Aligned || other.kind == Aligned)
      return aligned();
    return none();
  }
};
} // namespace

/// Could there be any read or write in effectsA on a variable written to in
/// effectsB?
static ConflictKind
anyRAWorWAW(llvm::ArrayRef<hlfir::DetailedEffectInstance> effectsA,
            llvm::ArrayRef<hlfir::DetailedEffectInstance> effectsB,
            fir::AliasAnalysis &aliasAnalysis) {
  ConflictKind result = ConflictKind::none();
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
                logConflict(llvm::dbgs(), writtenOrReadVarA, writtenVarB));
            return ConflictKind::any(); // unknown conflict.
          }
          writtenOrReadVarA = getStorageSource(writtenOrReadVarA);
          if (!aliasAnalysis.alias(writtenOrReadVarA, writtenVarB).isNo()) {
            mlir::Value arrayA = effectA.getOrderedElementalEffectOn();
            mlir::Value arrayB = effectB.getOrderedElementalEffectOn();
            if (arrayA && arrayB) {
              if (arrayA == arrayB) {
                result = result || ConflictKind::aligned();
                LLVM_DEBUG(logConflict(llvm::dbgs(), writtenOrReadVarA,
                                       writtenVarB, /*isAligned=*/true));
                continue;
              }
              auto overlap = fir::ArraySectionAnalyzer::analyze(arrayA, arrayB);
              if (overlap == fir::ArraySectionAnalyzer::SlicesOverlapKind::
                                 DefinitelyDisjoint)
                continue;
              if (overlap == fir::ArraySectionAnalyzer::SlicesOverlapKind::
                                 DefinitelyIdentical ||
                  overlap == fir::ArraySectionAnalyzer::SlicesOverlapKind::
                                 EitherIdenticalOrDisjoint) {
                result = result || ConflictKind::aligned();
                LLVM_DEBUG(logConflict(llvm::dbgs(), writtenOrReadVarA,
                                       writtenVarB, /*isAligned=*/true));
                continue;
              }
              LLVM_DEBUG(llvm::dbgs() << "conflicting arrays:" << arrayA
                                      << " and " << arrayB << "\n");
              return ConflictKind::any();
            }
            LLVM_DEBUG(
                logConflict(llvm::dbgs(), writtenOrReadVarA, writtenVarB));
            return ConflictKind::any();
          }
        }
    }
  return result;
}

/// Could there be any read or write in effectsA on a variable written to in
/// effectsB, or any read in effectsB on a variable written to in effectsA?
static ConflictKind
conflict(llvm::ArrayRef<hlfir::DetailedEffectInstance> effectsA,
         llvm::ArrayRef<hlfir::DetailedEffectInstance> effectsB) {
  fir::AliasAnalysis aliasAnalysis;
  // (RAW || WAW) || (WAR || WAW).
  ConflictKind result = anyRAWorWAW(effectsA, effectsB, aliasAnalysis);
  if (result.isAny())
    return result;
  return result || anyRAWorWAW(effectsB, effectsA, aliasAnalysis);
}

/// Could there be any write effects in "effects" affecting memory storages
/// that are not local to the current region.
static bool
anyNonLocalWrite(llvm::ArrayRef<hlfir::DetailedEffectInstance> effects,
                 mlir::Region &region) {
  return llvm::any_of(
      effects, [&region](const hlfir::DetailedEffectInstance &effect) {
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
  // This collection might be skipped, if there are no such parents,
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
  if (regionStates[&yieldRegion].saved)
    return;
  llvm::SmallVector<hlfir::DetailedEffectInstance> effects;
  gatherMemoryEffects(yieldRegion, leafRegionsMayOnlyRead, effects);
  // Yield has no effect as such, but in the context of order assignments.
  // The order assignments will usually read the yielded entity (except for
  // the yielded assignments LHS that is only read if this is an assignment
  // with a finalizer, or a user defined assignment where the LHS is
  // intent(inout)).
  if (yieldIsImplicitRead) {
    mlir::OpOperand *entity = getYieldedEntity(yieldRegion);
    if (entity && hlfir::isFortranVariableType(entity->get().getType())) {
      if (yieldRegion.getParentOfType<hlfir::ForallOp>())
        effects.emplace_back(mlir::MemoryEffects::Read::get(), entity);
      else
        effects.emplace_back(
            hlfir::DetailedEffectInstance::getArrayReadEffect(entity));
    }
  }
  if (!leafRegionsMayOnlyRead && anyNonLocalWrite(effects, yieldRegion)) {
    // Region with write effect must be executed only once (unless all writes
    // affect storages allocated inside the region): save it the first time it
    // is encountered.
    LLVM_DEBUG(llvm::dbgs()
                   << "saving eval because write effect prevents re-evaluation"
                   << "\n";);
    saveEvaluation(yieldRegion, effects, /*anyWrite=*/true);
  } else {
    ConflictKind conflictKind = conflict(effects, assignEffects);
    if (conflictKind.isAny()) {
      // Region that conflicts with the current assignments must be fully
      // evaluated and saved before doing the assignment (Note that it may
      // have already been evaluated without saving it before, but this
      // implies that it never conflicted with a prior assignment, so its value
      // should be the same.)
      saveEvaluation(yieldRegion, effects, /*anyWrite=*/false);
    } else {
      if (conflictKind.isAligned())
        pendingAlignedRegions.push_back(&yieldRegion);

      if (evaluationsMayConflict &&
          !conflict(effects, assignEvaluateEffects).isNone()) {
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
  }
}

void Scheduler::saveEvaluation(
    mlir::Region &yieldRegion,
    llvm::ArrayRef<hlfir::DetailedEffectInstance> effects, bool anyWrite) {
  savedAnyRegionForCurrentAssignment = true;
  auto &state = regionStates[&yieldRegion];
  if (state.modifiedInRun) {
    // The region was modified in a previous run, but we now realize we need its
    // value. We must save it before that modification run.
    auto &newRun = *schedule.emplace(*state.modifiedInRun, hlfir::Run{});
    newRun.actions.emplace_back(hlfir::SaveEntity{&yieldRegion});
    // We do not have the parent effects from that time easily available here.
    // However, since we are saving a parent of the current assignment, its
    // parents are also parents of the current assignment.
    newRun.memoryEffects.append(parentEvaluationEffects.begin(),
                                parentEvaluationEffects.end());
    newRun.memoryEffects.append(effects.begin(), effects.end());
    state.saved = true;
    LLVM_DEBUG(
        logSaveEvaluation(llvm::dbgs(), /*runid=*/0, yieldRegion, anyWrite););
    return;
  }

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
  state.saved = true;
  LLVM_DEBUG(
      logSaveEvaluation(llvm::dbgs(), schedule.size(), yieldRegion, anyWrite););
}

bool Scheduler::canFuseAssignmentWithPreviousRun() {
  // If a region was saved for the current assignment, the previous
  // run is already known to conflict. Skip the analysis.
  if (savedAnyRegionForCurrentAssignment || schedule.empty())
    return false;
  auto &previousRunEffects = schedule.back().memoryEffects;
  return !conflict(previousRunEffects, assignEffects).isAny() &&
         !conflict(previousRunEffects, parentEvaluationEffects).isAny() &&
         !conflict(previousRunEffects, independentEvaluationEffects).isAny();
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
// reverse of their execution order).
static void gatherAssignmentParents(
    hlfir::RegionAssignOp assign,
    llvm::SmallVectorImpl<hlfir::OrderedAssignmentTreeOpInterface> &parents) {
  gatherParents(mlir::cast<hlfir::OrderedAssignmentTreeOpInterface>(
                    assign.getOperation()),
                parents);
}

void Scheduler::finishSchedulingAssignment(hlfir::RegionAssignOp assign,
                                           bool leafRegionsMayOnlyRead) {
  // Schedule the assignment in a new run, unless it can be fused with the
  // previous run (if enabled and proven safe).
  currentRunIsReadOnly = false;
  bool fuse = tryFusingAssignments && canFuseAssignmentWithPreviousRun();
  if (!fuse) {
    // If we cannot fuse, we are about to start a new run.
    // Check if any parent region was modified in a previous run and needs to be
    // saved.
    llvm::SmallVector<hlfir::OrderedAssignmentTreeOpInterface> parents;
    gatherAssignmentParents(assign, parents);
    for (auto parent : parents) {
      llvm::SmallVector<mlir::Region *, 4> yieldRegions;
      parent.getLeafRegions(yieldRegions);
      for (mlir::Region *yieldRegion : yieldRegions) {
        if (regionStates[yieldRegion].modifiedInRun &&
            !regionStates[yieldRegion].saved) {
          LLVM_DEBUG(logRetroactiveSave(
              llvm::dbgs(), *yieldRegion,
              **regionStates[yieldRegion].modifiedInRun, assign));
          llvm::SmallVector<hlfir::DetailedEffectInstance> effects;
          gatherMemoryEffects(*yieldRegion, leafRegionsMayOnlyRead, effects);
          saveEvaluation(*yieldRegion, effects,
                         anyNonLocalWrite(effects, *yieldRegion));
        }
      }
    }
    schedule.emplace_back(hlfir::Run{});
  }

  // Mark pending aligned regions as modified in the current run (which is the
  // last one).
  auto runIt = std::prev(schedule.end());
  for (mlir::Region *region : pendingAlignedRegions)
    if (!regionStates[region].saved)
      regionStates[region].modifiedInRun = runIt;
  pendingAlignedRegions.clear();

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
    // During pointer assignments, the RHS data is not read, only the address
    // is taken.
    scheduler.startIndependentEvaluationGroup();
    scheduler.saveEvaluationIfConflict(
        assign.getRhsRegion(), leafRegionsMayOnlyRead,
        /*yieldIsImplicitRead=*/!assign.isPointerAssignment());
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
    scheduler.finishSchedulingAssignment(assign, leafRegionsMayOnlyRead);
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

[[maybe_unused]] static void
logRetroactiveSave(llvm::raw_ostream &os, mlir::Region &yieldRegion,
                   hlfir::Run &modifyingRun,
                   hlfir::RegionAssignOp currentAssign) {
  printRegionPath(os, yieldRegion) << " is modified in order by ";
  bool first = true;
  for (auto &action : modifyingRun.actions) {
    if (auto *assign = std::get_if<hlfir::RegionAssignOp>(&action)) {
      if (!first)
        os << ", ";
      printNodePath(os, assign->getOperation());
      first = false;
    }
  }
  os << " and is needed by ";
  printNodePath(os, currentAssign.getOperation());
  os << " that is scheduled in a later run\n";
}

[[maybe_unused]] static void logSaveEvaluation(llvm::raw_ostream &os,
                                               unsigned runid,
                                               mlir::Region &yieldRegion,
                                               bool anyWrite) {
  os << "run " << runid << " save  " << (anyWrite ? "(w)" : "  ") << ": ";
  printRegionPath(os, yieldRegion) << "\n";
}

[[maybe_unused]] static void
logAssignmentEvaluation(llvm::raw_ostream &os, unsigned runid,
                        hlfir::RegionAssignOp assign) {
  os << "run " << runid << " evaluate: ";
  printNodePath(os, assign.getOperation()) << "\n";
}

[[maybe_unused]] static void logConflict(llvm::raw_ostream &os,
                                         mlir::Value writtenOrReadVarA,
                                         mlir::Value writtenVarB,
                                         bool isAligned) {
  auto printIfValue = [&](mlir::Value var) -> llvm::raw_ostream & {
    if (!var)
      return os << "<unknown>";
    return os << var;
  };
  os << "conflict" << (isAligned ? " (aligned)" : "") << ": R/W: ";
  printIfValue(writtenOrReadVarA) << " W:";
  printIfValue(writtenVarB) << "\n";
}

[[maybe_unused]] static void
logStartScheduling(llvm::raw_ostream &os,
                   hlfir::OrderedAssignmentTreeOpInterface root) {
  os << "------------ scheduling ";
  printNodePath(os, root.getOperation());
  if (auto funcOp = root->getParentOfType<mlir::func::FuncOp>())
    os << " in " << funcOp.getSymName() << " ";
  os << "------------\n";
}

[[maybe_unused]] static void
logIfUnknownEffectValue(llvm::raw_ostream &os,
                        mlir::MemoryEffects::EffectInstance effect,
                        mlir::Operation &op) {
  if (effect.getValue() != nullptr)
    return;
  os << "unknown effected value (";
  os << (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()) ? "R" : "W");
  os << "): " << op << "\n";
}
