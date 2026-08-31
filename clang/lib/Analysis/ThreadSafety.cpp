//===- ThreadSafety.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A intra-procedural analysis for thread safety (e.g. deadlocks and race
// conditions), based off of an annotation system.
//
// See http://clang.llvm.org/docs/ThreadSafetyAnalysis.html
// for more information.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/ThreadSafety.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/Analyses/ThreadSafetyCommon.h"
#include "clang/Analysis/Analyses/ThreadSafetyTIL.h"
#include "clang/Analysis/Analyses/ThreadSafetyUtil.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TrailingObjects.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace threadSafety;

// Key method definition
ThreadSafetyHandler::~ThreadSafetyHandler() = default;

/// Whether \p LEK diagnoses the end of the function rather than an interior
/// join.
static bool isEndOfFunctionLEK(LockErrorKind LEK) {
  return LEK == LEK_LockedAtEndOfFunction ||
         LEK == LEK_NotLockedAtEndOfFunction;
}

/// True if capability attributes on \p Param describe the function reached
/// through it rather than the argument bound to it.
///
/// Sema accepts capability attributes on a parameter for two unrelated
/// purposes: a scoped-lockable parameter, where the attributes describe the
/// locks the passed scope object holds, and a parameter naming a function to
/// call -- a function pointer or a function reference -- where they describe
/// the requirements of the function called through it.
static bool isCallbackParam(const ParmVarDecl *Param) {
  QualType T = Param->getType().getNonReferenceType();
  return T->isFunctionPointerType() || T->isFunctionType();
}

/// Issue a warning about an invalid lock expression
static void warnInvalidLock(ThreadSafetyHandler &Handler,
                            const Expr *MutexExp, const NamedDecl *D,
                            const Expr *DeclExp, StringRef Kind) {
  SourceLocation Loc;
  if (DeclExp)
    Loc = DeclExp->getExprLoc();

  // FIXME: add a note about the attribute location in MutexExp or D
  if (Loc.isValid())
    Handler.handleInvalidLockExp(Loc);
}

namespace {

/// A set of CapabilityExpr objects, which are compiled from thread safety
/// attributes on a function.
class CapExprSet : public SmallVector<CapabilityExpr, 4> {
public:
  /// Whether M is in the list.
  bool contains(const CapabilityExpr &CapE) const {
    return llvm::any_of(
        *this, [&](const CapabilityExpr &CapE2) { return CapE.equals(CapE2); });
  }

  /// Push M onto list, but discard duplicates.
  void push_back_nodup(const CapabilityExpr &CapE) {
    if (!contains(CapE))
      push_back(CapE);
  }
};

class FactManager;
class FactSet;
class LockableFactEntry;

/// This is a helper class that stores a fact that is known at a
/// particular point in program execution. Concretely, a fact is a capability,
/// along with additional information, such as where it was acquired, whether
/// it is exclusive or shared, etc.
///
/// A capability's facts come in two forms, and a FactSet keeps them apart:
///
///  * At most one definite fact per capability: the capability is held (or,
///    for a negative capability, provably not held). A LockableFactEntry for
///    a mutex; a ScopedLockableFactEntry for a scope object's own capability,
///    which also lists the mutexes the object manages. Its reentrancy depth
///    counts the levels acquired on top of the first; the levels are
///    interchangeable -- releasing "a" level is releasing "the" level -- so
///    a counter is all they need.
///
///  * Any number of try facts per capability (TryFactEntry), one per
///    originating try-acquire call and lock kind: what that call's stored
///    result says about the capability. An unresolved try fact is conditional
///    -- "held if that call succeeded" -- and a branch on the call's result
///    resolves exactly the try facts that name it as their origin. A resolved
///    try fact stays behind as the record of what the branch proved, for
///    joins and rebranching.
///
/// Either form also records its source (SourceKind): whether the level was
/// acquired directly, asserted, declared on entry, or is managed by a scope
/// object, which is then the one to release it. The source is one value per
/// fact, not per level, so a definite fact whose levels came from different
/// sources carries only the first's.
///
/// A capability is:
///  - held if a definite fact exists,
///  - may be held if only conditional try facts exist,
///  - and not held if neither does.
///
/// The forms compose: a try-acquire over a held capability -- whatever its
/// reentrancy, since at runtime such a call fails rather than deadlocks --
/// adds a conditional try fact beside the definite fact, and a reentrant
/// acquire over a conditionally held capability adds the definite fact beside
/// the try facts. A branch's success edge folds the resolved try fact into the
/// definite fact (one level deeper, or created) and keeps the try fact as the
/// proof of that level, the failure edge drops it, and a release unwinds the
/// definite fact one level.
///
/// Branches are resolved in getEdgeLockset(). A resolved try fact stays
/// beside the fact it resolved to, so later branches on the same result
/// re-resolve it (an edge contradicting it is infeasible) and the join
/// demotion above can identify the hold its call proved. A proof one-sided
/// at a join is dropped: the merged hold is not determined by that call.
///
/// Both the join demotion and a branch's resolution rest on the premise
/// that a path not holding the capability carries a falsy stored result.
/// Released try facts police it: releasing a hold a call's success had proved
/// turns the call's try fact Released -- the result stays truthy, the hold is
/// gone -- and so does an unconditional release of a merely conditional
/// hold, for every conditional try fact of the capability. A join refuses
/// to demote-and-carry a hold across its call's stale result, and a
/// branch's success edge re-materializes a hold the analysis lost at a
/// join (e.g. around a loop) only when no surviving try fact, and no
/// definite fact of the inverse capability, contradicts it.
///
/// Conditionally held means "held if the try-acquire succeeded", so it warns
/// wherever a definite state is required: it does not satisfy capability
/// requirements, it violates exclusions and negative requirements,
/// releasing it warns (may not be held), and a blocking acquire of it
/// warns (may already be held). Asserts and same-kind reentrant acquires are
/// exempt from the acquire warning: they legitimately acquire a
/// possibly-held capability. An acquire of the other kind (shared vs.
/// exclusive) warns even for a reentrant capability: reentrancy nests
/// levels of one kind. Two unresolved try-acquires of one capability are
/// tracked as two try facts, each resolved by the branch on its own result.
/// A try fact is unique per call, so the same call executing again while its
/// own try fact is still unresolved has no second try fact to record the new
/// result in; it is diagnosed at the call. The walk order does not seem to
/// produce this case, and the check exists to keep that uniqueness honest.
///
/// When the analysis loses track of a conditionally held try fact -- at a join
/// with a path that does not hold it, or at the end of the function -- the
/// try-acquire result was never checked and the capability may be leaked.
class FactEntry : public CapabilityExpr {
public:
  enum FactEntryKind { Lockable, ScopedLockable, TryFact };

  /// Where a fact comes from.
  enum SourceKind {
    Acquired, ///< The fact has been directly acquired.
    Asserted, ///< The fact has been asserted to be held.
    Declared, ///< The fact is assumed to be held by callers.
    Managed,  ///< The fact has been acquired through a scoped capability.
  };

private:
  const FactEntryKind Kind : 8;

  /// Exclusive or shared.
  LockKind LKind : 8;

  /// How it was acquired.
  SourceKind Source : 8;

  /// Where it was acquired.
  SourceLocation AcquireLoc;

protected:
  ~FactEntry() = default;

public:
  FactEntry(FactEntryKind FK, const CapabilityExpr &CE, LockKind LK,
            SourceLocation Loc, SourceKind Src)
      : CapabilityExpr(CE), Kind(FK), LKind(LK), Source(Src), AcquireLoc(Loc) {}

  LockKind kind() const { return LKind;      }
  SourceLocation loc() const { return AcquireLoc; }
  FactEntryKind getFactEntryKind() const { return Kind; }

  SourceKind source() const { return Source; }
  bool asserted() const { return Source == Asserted; }
  bool declared() const { return Source == Declared; }
  bool managed() const { return Source == Managed; }

  /// Whether losing track of this fact warrants a diagnostic: an asserted
  /// or universal capability's hold is not something the analyzed code is
  /// expected to release, and a negative fact is not a hold at all.
  bool lossNeedsWarning() const {
    return !asserted() && !negative() && !isUniversal();
  }

  virtual void
  handleRemovalFromIntersection(const FactSet &FSet, FactManager &FactMan,
                                SourceLocation JoinLoc, LockErrorKind LEK,
                                ThreadSafetyHandler &Handler) const = 0;
  virtual void handleLock(FactSet &FSet, FactManager &FactMan,
                          const FactEntry &entry,
                          ThreadSafetyHandler &Handler) const = 0;
  virtual void handleUnlock(FactSet &FSet, FactManager &FactMan,
                            const CapabilityExpr &Cp, SourceLocation UnlockLoc,
                            bool FullyRemove,
                            ThreadSafetyHandler &Handler) const = 0;

  // Return true if LKind >= LK, where exclusive > shared
  bool isAtLeast(LockKind LK) const {
    return  (LKind == LK_Exclusive) || (LK == LK_Shared);
  }
};

/// The conditional fact of one try-acquire call's stored result: created at the
/// call, resolved by the branches on the result, and kept afterwards as the
/// record of what they proved, in one of these states:
///
///   Conditional    The result is unknown here: the capability is held iff it
///                  is truthy. The only state that is a (possible) hold.
///   ProvedHeld     The result is truthy on every path here, and the hold it
///                  proved is live: one level of the definite fact comes from
///                  branching on this call.
///   ProvedNotHeld  The result is falsy on every path here: the call acquired
///                  nothing, and its branch's success edge is infeasible.
///   Released       The result is truthy on some path here, but the hold it
///                  proved was released since: a branch on the stale result
///                  must not resurrect it.
///
/// Two try facts of one identity meet only at a join.
class TryFactEntry final : public FactEntry {
public:
  enum class State : uint8_t {
    Conditional,
    ProvedHeld,
    ProvedNotHeld,
    Released
  };

private:
  /// The try-acquire call whose result created this fact state.
  const Expr *Origin;

  State St : 8;

  /// For a ProvedNotHeld try fact: the failure edge belonged to a non-void `?:`
  /// terminator, i.e. the arms of a value merge were split. Their join
  /// reconstitutes the conditional try fact silently even without
  /// -Wthread-safety-beta: the branch was never honored before, so its
  /// join never warned.
  ///
  /// This is an annotation on one edge's fact rather than a state of the
  /// lattice below: the join reads it, and no transition carries it, so two
  /// facts whose marks differ meet as an unmarked fact of the meet's state.
  bool ArmSplit : 1;

  /// For a Conditional try fact: the hold was released on some path into
  /// here -- the try fact met the same call's Released try fact at a join, or
  /// a loop body's release reached the loop head's exit set. The hold it may
  /// still prove on the other paths cannot be resolved: a branch on the
  /// result must not resurrect the released hold, so the success edge
  /// marks the try fact Released rather than promoting it. It may still be
  /// held, and is diagnosed like any conditional try fact.
  bool MayBeReleased : 1;

  /// For a Released try fact, or a Conditional one marked MayBeReleased: where
  /// the hold was released (the note of a later unmatched unlock).
  SourceLocation ReleaseLoc;

  TryFactEntry(const CapabilityExpr &CE, LockKind LK, SourceLocation Loc,
               SourceKind Src, const Expr *Origin)
      : FactEntry(TryFact, CE, LK, Loc, Src), Origin(Origin),
        St(State::Conditional), ArmSplit(false), MayBeReleased(false) {
    assert(Origin && "a try fact speaks about a call");
  }

public:
  static TryFactEntry *create(llvm::BumpPtrAllocator &Alloc,
                              const TryFactEntry &Other) {
    return new (Alloc) TryFactEntry(Other);
  }

  static TryFactEntry *create(llvm::BumpPtrAllocator &Alloc,
                              const CapabilityExpr &CE, LockKind LK,
                              SourceLocation Loc, SourceKind Src,
                              const Expr *Origin) {
    return new (Alloc) TryFactEntry(CE, LK, Loc, Src, Origin);
  }

  const Expr *origin() const { return Origin; }
  State state() const { return St; }
  bool conditional() const { return St == State::Conditional; }
  bool provedHeld() const { return St == State::ProvedHeld; }
  bool provedNotHeld() const { return St == State::ProvedNotHeld; }
  bool released() const { return St == State::Released; }
  bool armSplit() const { return ArmSplit; }
  bool mayBeReleased() const { return MayBeReleased; }
  SourceLocation releaseLoc() const { return ReleaseLoc; }

  /// This try fact in state \p S, without marks: a ProvedNotHeld try fact's
  /// `?:` split mark and a Conditional try fact's MayBeReleased mark are kept
  /// only while the state stays. The fact itself if that is already its state.
  const TryFactEntry *withState(FactManager &FactMan, State S) const;
  /// This ProvedNotHeld try fact with its `?:` split mark set.
  const TryFactEntry *withArmSplit(FactManager &FactMan) const;
  /// This try fact released by a release at \p Loc.
  const TryFactEntry *asReleased(FactManager &FactMan,
                                 SourceLocation Loc) const;
  /// This Conditional try fact with its MayBeReleased mark set, the
  /// release at \p Loc.
  const TryFactEntry *asMayBeReleased(FactManager &FactMan,
                                      SourceLocation Loc) const;

  /// The definite fact this try fact's success proves: the same
  /// acquisition -- kind, source and location -- now held.
  const LockableFactEntry *asDefinite(FactManager &FactMan) const;

  /// The state two try facts of one identity meet in, one on each side of a
  /// join point. Released on either side is released -- a stale result is
  /// stale however it got here; two equal states meet in themselves; and
  /// anything else meets in Conditional, since a proof about one side's
  /// paths is no proof about the other's.
  ///
  /// This is the meet of the states, not the outcome of the join. What the
  /// merged fact becomes, and what happens to the definite facts a
  /// ProvedHeld side proved, is LocksetJoin::joinTryFactPair()'s to decide,
  /// and for the mixed cells it mostly decides something else: the
  /// reconstitution of "held iff C" from a ProvedHeld and a ProvedNotHeld
  /// side is not taken there, and a Conditional side meeting a Released one
  /// may keep its possible hold, marked may-be-released, instead.
  static State joinStates(State A, State B) {
    if (A == State::Released || B == State::Released)
      return State::Released;
    if (A == B)
      return A;
    return State::Conditional;
  }

  /// A Conditional try fact meeting the Released try fact of its call keeps
  /// its possible hold only where the join would keep a one-sided
  /// conditional try fact, and then as a may-be-released one (see \c
  /// MayBeReleased): intersectAndWarn() decides that, over the lattice.
  ///
  /// A try fact is never dispatched through the one-fact-per-capability
  /// protocol below: the analyzer orchestrates its transitions itself.
  void
  handleRemovalFromIntersection(const FactSet &FSet, FactManager &FactMan,
                                SourceLocation JoinLoc, LockErrorKind LEK,
                                ThreadSafetyHandler &Handler) const override {
    llvm_unreachable("a try fact is lost through its own join rules");
  }

  void handleLock(FactSet &FSet, FactManager &FactMan, const FactEntry &entry,
                  ThreadSafetyHandler &Handler) const override {
    llvm_unreachable("a try fact is not a hold to reacquire");
  }

  void handleUnlock(FactSet &FSet, FactManager &FactMan,
                    const CapabilityExpr &Cp, SourceLocation UnlockLoc,
                    bool FullyRemove,
                    ThreadSafetyHandler &Handler) const override {
    llvm_unreachable("a try fact is not a hold to release");
  }

  static bool classof(const FactEntry *A) {
    return A->getFactEntryKind() == TryFact;
  }
};

using FactID = unsigned short;

/// FactManager manages the memory for all facts that are created during
/// the analysis of a single routine.
class FactManager {
private:
  llvm::BumpPtrAllocator &Alloc;
  std::vector<const FactEntry *> Facts;

public:
  FactManager(llvm::BumpPtrAllocator &Alloc) : Alloc(Alloc) {}

  template <typename T, typename... ArgTypes>
  T *createFact(ArgTypes &&...Args) {
    static_assert(std::is_trivially_destructible_v<T>);
    return T::create(Alloc, std::forward<ArgTypes>(Args)...);
  }

  FactID newFact(const FactEntry *Entry) {
    Facts.push_back(Entry);
    assert(Facts.size() - 1 <= std::numeric_limits<FactID>::max() &&
           "FactID space exhausted");
    return static_cast<unsigned short>(Facts.size() - 1);
  }

  const FactEntry &operator[](FactID F) const { return *Facts[F]; }

  /// Whether \p Loc is the location of a try-acquire call that names \p Cp:
  /// a negative fact for \p Cp sitting there was recorded by a branch on the
  /// call's result (getEdgeLockset()), not by a release. The capability is
  /// part of the question because one call can do both -- release one
  /// capability and try-acquire another -- and the release still deserves
  /// its note. Either polarity counts: a try-acquire of a negative
  /// capability records the failure of a release the same way.
  bool isTryAcquireLoc(SourceLocation Loc, const CapabilityExpr &Cp) const {
    auto It = TryAcquireLocs.find(Loc);
    return It != TryAcquireLocs.end() &&
           (It->second.contains(Cp) || It->second.contains(!Cp));
  }
  void addTryAcquireLoc(SourceLocation Loc, const CapabilityExpr &Cp) {
    TryAcquireLocs[Loc].push_back_nodup(Cp);
  }

  /// Whether an unconditional acquire, assert or release has consumed a
  /// conditional try fact of the call \p Origin. The call's stored result
  /// then no longer determines a level of its own -- the definite fact that
  /// took its place does -- so a branch on that result must not
  /// re-materialize a hold once the definite fact is gone in its turn
  /// (getEdgeLockset()). Recorded per call rather than per capability and
  /// path: like the tracked capabilities themselves it is a property of the
  /// walk, not of a fact set, and refusing a re-materialization is the
  /// conservative side of it.
  bool spentTryAcquire(const Expr *Origin) const {
    return SpentTryAcquires.contains(Origin);
  }
  void addSpentTryAcquire(const Expr *Origin) {
    SpentTryAcquires.insert(Origin);
  }
  /// A fresh execution of \p Origin overwrites its stored result, so
  /// whatever spent the last one no longer speaks about this one.
  void clearSpentTryAcquire(const Expr *Origin) {
    SpentTryAcquires.erase(Origin);
  }

  /// Whether a join has lost \p Origin's conditional try fact without the
  /// result being checked -- what the beta diagnostic reports, recorded
  /// whether or not it is enabled. The call's outcome is then unaccounted
  /// for, so a merge of its result with another call's cannot speak for
  /// both (decodeTrylockCond()).
  bool lostUnchecked(const Expr *Origin) const {
    return LostUnchecked.contains(Origin);
  }
  void addLostUnchecked(const Expr *Origin) { LostUnchecked.insert(Origin); }

  /// Whether \p Origin ever executed while another call's try fact of the
  /// same capability was still unresolved: the two calls' holds can then
  /// coexist, so neither's result speaks for the other and a merge of them
  /// is not a twin (decodeTrylockCond()). In the retry idiom each call runs
  /// only after the previous result was checked false, so this never
  /// happens; in the retry-without-checking shape it always does.
  bool coexecutedTryAcquire(const Expr *Origin) const {
    return Coexecuted.contains(Origin);
  }
  void addCoexecutedTryAcquire(const Expr *Origin) {
    Coexecuted.insert(Origin);
  }

private:
  llvm::SmallDenseMap<SourceLocation, CapExprSet, 4> TryAcquireLocs;
  llvm::SmallDenseSet<const Expr *, 4> SpentTryAcquires;
  llvm::SmallDenseSet<const Expr *, 4> LostUnchecked;
  llvm::SmallDenseSet<const Expr *, 4> Coexecuted;
};

inline const TryFactEntry *TryFactEntry::withState(FactManager &FactMan,
                                                   State S) const {
  if (S == St && !ArmSplit && !MayBeReleased)
    return this;
  auto *NewFact = FactMan.createFact<TryFactEntry>(*this);
  NewFact->St = S;
  NewFact->ArmSplit = false;
  NewFact->MayBeReleased = false;
  return NewFact;
}

const TryFactEntry *TryFactEntry::withArmSplit(FactManager &FactMan) const {
  assert(provedNotHeld() && "only a failure edge splits a value merge");
  if (ArmSplit)
    return this;
  auto *NewFact = FactMan.createFact<TryFactEntry>(*this);
  NewFact->ArmSplit = true;
  return NewFact;
}

const TryFactEntry *TryFactEntry::asReleased(FactManager &FactMan,
                                             SourceLocation Loc) const {
  auto *NewFact = FactMan.createFact<TryFactEntry>(*this);
  NewFact->St = State::Released;
  NewFact->ArmSplit = false;
  NewFact->MayBeReleased = false;
  NewFact->ReleaseLoc = Loc;
  return NewFact;
}

const TryFactEntry *TryFactEntry::asMayBeReleased(FactManager &FactMan,
                                                  SourceLocation Loc) const {
  assert(conditional() && "only a possible hold is marked MayBeReleased");
  auto *NewFact = FactMan.createFact<TryFactEntry>(*this);
  NewFact->MayBeReleased = true;
  NewFact->ReleaseLoc = Loc;
  return NewFact;
}

/// A FactSet is the set of facts that are known to be true at a
/// particular program point.  FactSets must be small, because they are
/// frequently copied, and are thus implemented as a set of indices into a
/// table maintained by a FactManager.  A typical FactSet only holds 1 or 2
/// locks, so we can get away with doing a linear search for lookup.  Note
/// that a hashtable or map is inappropriate in this case, because lookups
/// may involve partial pattern matches, rather than exact matches.
///
/// A capability may be represented by several facts at once:
/// at most one definite fact, and any number of other facts, unique per
/// originating call and lock kind. The accessors name which sort of fact they
/// will look for; there is no lookup for "the" fact of a capability.
class FactSet {
private:
  using FactVec = SmallVector<FactID, 4>;

  FactVec FactIDs;

public:
  using iterator = FactVec::iterator;
  using const_iterator = FactVec::const_iterator;

private:
  template <typename Pred> iterator findIf(FactManager &FM, Pred P) {
    return llvm::find_if(*this, [&](FactID ID) { return P(FM[ID]); });
  }
  template <typename Pred>
  const FactEntry *findEntry(FactManager &FM, Pred P) const {
    auto I = llvm::find_if(*this, [&](FactID ID) { return P(FM[ID]); });
    return I != end() ? &FM[*I] : nullptr;
  }

  /// Whether \p FE is a try fact of \p CapE, unresolved: the capability may
  /// be held through it.
  static bool isConditionalOf(const FactEntry &FE, const CapabilityExpr &CapE) {
    const auto *W = dyn_cast<TryFactEntry>(&FE);
    return W && W->conditional() && W->matches(CapE);
  }

  /// Whether \p FE is the try fact of \p CapE from the call \p Origin in
  /// lock kind \p Kind, whatever its state: a try fact's identity.
  static bool isTryFactOf(const FactEntry &FE, const CapabilityExpr &CapE,
                          const Expr *Origin, LockKind Kind) {
    const auto *W = dyn_cast<TryFactEntry>(&FE);
    return W && W->origin() == Origin && W->kind() == Kind && W->matches(CapE);
  }

public:
  iterator begin() { return FactIDs.begin(); }
  const_iterator begin() const { return FactIDs.begin(); }

  iterator end() { return FactIDs.end(); }
  const_iterator end() const { return FactIDs.end(); }

  bool isEmpty() const { return FactIDs.size() == 0; }

  // Return true if the set holds no definite positive capability. It may
  // hold negative facts or try facts, unlike isEmpty, which tests the set
  // itself.
  bool holdsNoCapability(FactManager &FactMan) const {
    for (const auto FID : *this) {
      if (!FactMan[FID].negative() && isDefinite(FactMan[FID]))
        return false;
    }
    return true;
  }

  void addLockByID(FactID ID) { FactIDs.push_back(ID); }

  /// Whether \p FE states that a capability is held, rather than speaking
  /// about the result of a try-acquire.
  static bool isDefinite(const FactEntry &FE) { return !isa<TryFactEntry>(FE); }

  /// Add \p Entry to the set. The walk keeps at most one definite fact of a
  /// capability in a set, while a capability may have any number of try facts
  /// -- one per originating call and lock kind, which is the invariant a try
  /// fact relaxes, and which its identity must respect.
  FactID addLock(FactManager &FM, const FactEntry *Entry) {
    assert((!isa<TryFactEntry>(Entry) ||
            !findTryFact(FM, *Entry, cast<TryFactEntry>(Entry)->origin(),
                         Entry->kind())) &&
           "a try fact is identified by its capability, call and kind");
    FactID F = FM.newFact(Entry);
    FactIDs.push_back(F);
    return F;
  }

  /// Remove the fact at \p It, moving the set's last element into its
  /// slot.
  void erase(iterator It) {
    *It = FactIDs.back();
    FactIDs.pop_back();
  }

  /// \name Lookup by identity
  /// The fact \p F itself, which a caller obtained from a lookup below.
  /// \{
  iterator findFactIter(FactManager &FM, const FactEntry &F) {
    return findIf(FM, [&](const FactEntry &FE) { return &FE == &F; });
  }

  /// Erase what \p It denotes, if anything; return whether it did.
  bool removeAt(iterator It) {
    if (It == end())
      return false;
    erase(It);
    return true;
  }

  bool removeFact(FactManager &FM, const FactEntry &F) {
    return removeAt(findFactIter(FM, F));
  }

  std::optional<FactID> replaceFact(FactManager &FM, iterator It,
                                    const FactEntry *Entry) {
    if (It == end())
      return std::nullopt;
    FactID F = FM.newFact(Entry);
    *It = F;
    return F;
  }

  std::optional<FactID> replaceFact(FactManager &FM, const FactEntry &Old,
                                    const FactEntry *Entry) {
    return replaceFact(FM, findFactIter(FM, Old), Entry);
  }
  /// \}

  /// \name Definite facts
  /// The one fact stating that \p CapE is held (or, negated, not held).
  /// \{
  iterator findDefiniteIter(FactManager &FM, const CapabilityExpr &CapE) {
    return findIf(FM, [&](const FactEntry &FE) {
      return isDefinite(FE) && FE.matches(CapE);
    });
  }

  const FactEntry *findDefinite(FactManager &FM,
                                const CapabilityExpr &CapE) const {
    return findEntry(FM, [&](const FactEntry &FE) {
      return isDefinite(FE) && FE.matches(CapE);
    });
  }

  const FactEntry *findDefiniteUniv(FactManager &FM,
                                    const CapabilityExpr &CapE) const {
    return findEntry(FM, [&](const FactEntry &FE) {
      return isDefinite(FE) && FE.matchesUniv(CapE);
    });
  }

  const FactEntry *findDefinitePartialMatch(FactManager &FM,
                                            const CapabilityExpr &CapE) const {
    return findEntry(FM, [&](const FactEntry &FE) {
      return isDefinite(FE) && FE.partiallyMatches(CapE);
    });
  }

  bool removeDefinite(FactManager &FM, const CapabilityExpr &CapE) {
    return removeAt(findDefiniteIter(FM, CapE));
  }
  /// \}

  /// \name Try facts
  /// The try facts of \p CapE.
  /// \{
  iterator findTryFactIter(FactManager &FM, const CapabilityExpr &CapE,
                           const Expr *Origin, LockKind Kind) {
    return findIf(FM, [&](const FactEntry &FE) {
      return isTryFactOf(FE, CapE, Origin, Kind);
    });
  }

  const TryFactEntry *findTryFact(FactManager &FM, const CapabilityExpr &CapE,
                                  const Expr *Origin, LockKind Kind) const {
    return cast_or_null<TryFactEntry>(findEntry(FM, [&](const FactEntry &FE) {
      return isTryFactOf(FE, CapE, Origin, Kind);
    }));
  }

  /// The try fact of \p CapE from the call \p Origin in state \p S,
  /// whichever its kind: a call's failure record (ProvedNotHeld), the proof
  /// of a hold (ProvedHeld), its stale result (Released), or its unresolved
  /// try fact (Conditional).
  const TryFactEntry *findTryFactFrom(FactManager &FM,
                                      const CapabilityExpr &CapE,
                                      const Expr *Origin,
                                      TryFactEntry::State S) const {
    return findTryFactIf(FM, CapE, [&](const TryFactEntry &W) {
      return W.state() == S && W.origin() == Origin;
    });
  }

  /// The first unresolved try fact of \p CapE, whichever its origin: the
  /// capability may be held.
  const TryFactEntry *firstConditional(FactManager &FM,
                                       const CapabilityExpr &CapE) const {
    return cast_or_null<TryFactEntry>(findEntry(
        FM, [&](const FactEntry &FE) { return isConditionalOf(FE, CapE); }));
  }

  bool anyConditional(FactManager &FM, const CapabilityExpr &CapE) const {
    return firstConditional(FM, CapE) != nullptr;
  }

  /// Whether every unresolved try fact of \p CapE has the lock kind \p Kind.
  bool conditionalsAllOfKind(FactManager &FM, const CapabilityExpr &CapE,
                             LockKind Kind) const {
    return llvm::all_of(*this, [&](FactID ID) {
      return !isConditionalOf(FM[ID], CapE) || FM[ID].kind() == Kind;
    });
  }

  /// Collect every try fact of \p CapE, resolved or not, into \p Out, so
  /// that a caller can mutate the set while visiting them.
  void collectTryFacts(FactManager &FM, const CapabilityExpr &CapE,
                       SmallVectorImpl<const TryFactEntry *> &Out) const {
    for (FactID ID : *this)
      if (const auto *W = dyn_cast<TryFactEntry>(&FM[ID]);
          W && W->matches(CapE))
        Out.push_back(W);
  }

  /// Remove every unresolved try fact of \p CapE originating from
  /// \p Origin (one per lock kind); returns whether there was any.
  bool removeConditionalsOf(FactManager &FM, const CapabilityExpr &CapE,
                            const Expr *Origin) {
    const size_t Before = FactIDs.size();
    llvm::erase_if(FactIDs, [&](FactID ID) {
      if (!isConditionalOf(FM[ID], CapE) ||
          cast<TryFactEntry>(FM[ID]).origin() != Origin)
        return false;
      FM.addSpentTryAcquire(Origin);
      return true;
    });
    return FactIDs.size() != Before;
  }

  /// Remove every unresolved try fact of \p CapE: the capability is no
  /// longer possibly held through any of them.
  void removeAllConditional(FactManager &FM, const CapabilityExpr &CapE) {
    llvm::erase_if(FactIDs, [&](FactID ID) {
      if (!isConditionalOf(FM[ID], CapE))
        return false;
      FM.addSpentTryAcquire(cast<TryFactEntry>(FM[ID]).origin());
      return true;
    });
  }

  /// The hold of \p CapE is gone: every ProvedHeld try fact of it no longer
  /// proves a live level. Its call's stored result stays truthy while the
  /// hold is released -- released (TryFactEntry::State::Released) -- so that a
  /// later branch on it does not resurrect the hold.
  void releaseProved(FactManager &FM, const CapabilityExpr &CapE,
                     SourceLocation Loc) {
    for (FactID &ID : FactIDs) {
      const auto *W = dyn_cast<TryFactEntry>(&FM[ID]);
      if (W && W->provedHeld() && W->matches(CapE))
        ID = FM.newFact(W->asReleased(FM, Loc));
    }
  }

  /// A blocking acquisition of \p CapE: its stale results are retired. A
  /// Released try fact records that the hold its call proved is gone, which
  /// keeps a branch on the still-truthy result from resurrecting it; the
  /// capability is held again now, through an acquisition of its own, so
  /// there is no lost hold left to protect. A failure record is not a stale
  /// result and stays (failed_survives_lock).
  void retireReleased(FactManager &FM, const CapabilityExpr &CapE) {
    llvm::erase_if(FactIDs, [&](FactID ID) {
      const auto *W = dyn_cast<TryFactEntry>(&FM[ID]);
      return W && W->released() && W->matches(CapE);
    });
  }

  /// An unconditional release of \p CapE while it is only conditionally
  /// held: the release took a hold or aborted, so nothing is held
  /// afterwards and every conditional try fact's stored result is stale.
  void releaseConditional(FactManager &FM, const CapabilityExpr &CapE,
                          SourceLocation Loc) {
    for (FactID &ID : FactIDs) {
      const auto *W = dyn_cast<TryFactEntry>(&FM[ID]);
      if (W && W->conditional() && W->matches(CapE))
        ID = FM.newFact(W->asReleased(FM, Loc));
    }
  }

  /// The first try fact of \p CapE satisfying \p Pred, whichever its
  /// origin, kind or state.
  template <typename Pred>
  const TryFactEntry *findTryFactIf(FactManager &FM, const CapabilityExpr &CapE,
                                    Pred P) const {
    return cast_or_null<TryFactEntry>(findEntry(FM, [&](const FactEntry &FE) {
      const auto *W = dyn_cast<TryFactEntry>(&FE);
      return W && W->matches(CapE) && P(*W);
    }));
  }

  /// Whether a surviving fact of \p CapE refutes re-materializing a hold
  /// proved by the call \p Origin (getEdgeLockset()): a definite fact or an
  /// unresolved try fact means the hold was never lost, another call's
  /// ProvedNotHeld try fact that the capability was not acquired since. A
  /// Released try fact says a stored result went stale, which is evidence
  /// about its own call only -- so another call's does not refute this one,
  /// exactly as at a rebranch join
  /// (LocksetJoin::rebranchVetoedByReleased()).
  bool refutesHoldOf(FactManager &FM, const CapabilityExpr &CapE,
                     const Expr *Origin) const {
    return findEntry(FM, [&](const FactEntry &FE) {
             if (!FE.matches(CapE))
               return false;
             const auto *W = dyn_cast<TryFactEntry>(&FE);
             return !W || !W->released() || W->origin() == Origin;
           }) != nullptr;
  }
  /// \}

  /// The definite fact of \p CapE, or else its first unresolved try fact:
  /// whether the capability is held or may be held. A resolved try fact is
  /// neither.
  const FactEntry *findDefiniteOrConditional(FactManager &FM,
                                             const CapabilityExpr &CapE) const {
    return findEntry(FM, [&](const FactEntry &FE) {
      return isa<TryFactEntry>(FE) ? isConditionalOf(FE, CapE)
                                   : FE.matches(CapE);
    });
  }

  /// \name Counterparts
  /// The fact of the same form as \p F -- definite, or the try fact of the
  /// same call in the same kind -- for \p F's capability: what a join
  /// pairs \p F with.
  /// \{
  iterator findCounterpartIter(FactManager &FM, const FactEntry &F) {
    if (const auto *W = dyn_cast<TryFactEntry>(&F))
      return findTryFactIter(FM, F, W->origin(), W->kind());
    return findDefiniteIter(FM, F);
  }

  const FactEntry *findCounterpart(FactManager &FM, const FactEntry &F) const {
    if (const auto *W = dyn_cast<TryFactEntry>(&F))
      return findTryFact(FM, F, W->origin(), W->kind());
    return findDefinite(FM, F);
  }
  /// \}
  /// Whether a definite fact of the capability declared by \p Vd is present.
  bool containsMutexDecl(FactManager &FM, const ValueDecl *Vd) const {
    return llvm::any_of(*this, [&](FactID ID) {
      return !isa<TryFactEntry>(FM[ID]) && FM[ID].valueDecl() == Vd;
    });
  }
};

class ThreadSafetyAnalyzer;

} // namespace

namespace clang {
namespace threadSafety {

class BeforeSet {
private:
  using BeforeVect = SmallVector<const ValueDecl *, 4>;

  struct BeforeInfo {
    BeforeVect Vect;
    int Visited = 0;

    BeforeInfo() = default;
    BeforeInfo(BeforeInfo &&) = default;
  };

  using BeforeMap =
      llvm::DenseMap<const ValueDecl *, std::unique_ptr<BeforeInfo>>;
  using CycleMap = llvm::DenseMap<const ValueDecl *, bool>;

public:
  BeforeSet() = default;

  BeforeInfo* insertAttrExprs(const ValueDecl* Vd,
                              ThreadSafetyAnalyzer& Analyzer);

  BeforeInfo *getBeforeInfoForDecl(const ValueDecl *Vd,
                                   ThreadSafetyAnalyzer &Analyzer);

  void checkBeforeAfter(const ValueDecl* Vd,
                        const FactSet& FSet,
                        ThreadSafetyAnalyzer& Analyzer,
                        SourceLocation Loc, StringRef CapKind);

private:
  BeforeMap BMap;
  CycleMap CycMap;
};

} // namespace threadSafety
} // namespace clang

namespace {

class LocalVariableMap;

using LocalVarContext = llvm::ImmutableMap<const NamedDecl *, unsigned>;

/// A side (entry or exit) of a CFG node.
enum CFGBlockSide { CBS_Entry, CBS_Exit };

/// How one direction of a terminator's branch resolves one capability of the
/// branched-on try-acquire call.
enum class CapResolution : uint8_t {
  Unknown, ///< The direction does not decide this capability's outcome.
  Success, ///< The call acquired the capability.
  Failure, ///< The call did not acquire the capability.
};

/// One capability of a try-acquire call, with the kind it is acquired as and
/// the resolution one branch direction or edge proves for it.
struct TrylockEdgeCap {
  CapabilityExpr Cap;
  LockKind Kind;
  CapResolution Resolution;
};

/// The success-value profile of the attributes naming one capability
/// among those recorded for a try-acquire call: the polarities that
/// report acquisition, and the exact truthy success region -- the
/// recorded codes, or any nonzero result (AnyNonzero: a boolean success
/// value, or no recorded codes).
struct CapProfile {
  bool Truthy = false, Falsy = false, AnyNonzero = false;
  SmallVector<llvm::APSInt, 2> Codes;

  /// Whether pinning the result to the nonzero value \p V proves the
  /// acquisition: the value lies in the truthy success region.
  bool containsValue(const llvm::APSInt &V) const {
    return Truthy &&
           (AnyNonzero || llvm::any_of(Codes, [&](const llvm::APSInt &C) {
              return llvm::APSInt::isSameValue(C, V);
            }));
  }
  /// Whether the whole success region -- both polarities' -- is ruled
  /// out by \p Excluded (a predicate for one value), proving the
  /// capability was not acquired.
  bool regionExcludedBy(
      llvm::function_ref<bool(const llvm::APSInt &)> Excluded) const {
    return (!Truthy || (!AnyNonzero && llvm::all_of(Codes, Excluded))) &&
           (!Falsy || Excluded(llvm::APSInt::get(0)));
  }
};

/// What a terminator's branch proves about the capabilities of the try-acquire
/// call it branches on.
struct TrylockBranch {
  /// The try-acquire call whose result the terminator branches on, or null if
  /// it does not branch on one.
  const CallExpr *TrylockCall = nullptr;
  /// When the branched-on variable merges the results of two structurally
  /// identical try-acquire calls, the second path's call (TrylockCall
  /// resolves to the first path's); null otherwise. A join folds the two
  /// calls' try facts into the resolved call's (intersectAndWarn()).
  const CallExpr *MergedCall = nullptr;
  /// The call may not have executed on edges of this direction (the
  /// branched-on variable merges its result with a constant): each
  /// capability's resolution holds only if it did (TrylockEdge's
  /// Ambiguous).
  bool AmbiguousTrue = false, AmbiguousFalse = false;
  /// The decode reached the call through the right-hand side of a `&&` or
  /// `||` (TrylockDecode::ShortCircuit): a hold this branch's edge would
  /// re-materialize may belong to a path that short-circuited past the
  /// call, so no edge of this terminator manufactures one.
  bool ShortCircuit = false;
  /// The branched-on value is the call's result itself -- no negation
  /// or comparison in between -- so an exact value an edge carries (a
  /// switch case label, a default edge's exclusions) applies to the
  /// result and refines the capabilities' resolutions on that edge.
  bool ValueIsResult = false;
  /// Whether the per-direction resolutions below were decided by an
  /// exact value comparison (`result == code`) rather than truthiness.
  /// Such a conclusion does not survive an edge where the branched-on
  /// value may be a merge's constant (resolveTrylockEdge()).
  bool ValueCompared = false;
  /// The call's capabilities for each branch direction. Every capability
  /// recorded for the call appears in both lists, with the resolution that
  /// direction proves for it: Unknown where the direction decides nothing
  /// about it.
  SmallVector<TrylockEdgeCap, 1> OnTrue, OnFalse;
  /// Each capability's success-value profile, in the order of the two lists
  /// above: what an edge carrying an exact value resolves it against
  /// (resolveTrylockEdge()). Computed with the rest of the decode, since a
  /// terminator's edges all ask the same questions of it.
  SmallVector<CapProfile, 1> Profiles;
};

/// CFGBlockInfo is a struct which contains all the information that is
/// maintained for each block in the CFG.  See LocalVariableMap for more
/// information about the contexts.
struct CFGBlockInfo {
  // Lockset held at entry to block
  FactSet EntrySet;

  // Lockset held at exit from block
  FactSet ExitSet;

  // Context held at entry to block
  LocalVarContext EntryContext;

  // Context held at exit from block
  LocalVarContext ExitContext;

  // Location of first statement in block
  SourceLocation EntryLoc;

  // Location of last statement in block.
  SourceLocation ExitLoc;

  // Used to replay contexts later
  unsigned EntryIndex;

  // Is this block reachable?
  bool Reachable = false;

  // Memoized decode of this block's terminator branch.
  std::optional<TrylockBranch> TryBranch;

  // Whether the block is reachable only through infeasible edges (or
  // through other such blocks): analyzed for coverage -- diagnostics
  // inside it are real -- but its exit set carries provably dead state,
  // which downstream joins must not consume as if it were a live path
  // (runAnalysis()).
  bool CoverageOnly = false;

  const FactSet &getSet(CFGBlockSide Side) const {
    return Side == CBS_Entry ? EntrySet : ExitSet;
  }

  SourceLocation getLocation(CFGBlockSide Side) const {
    return Side == CBS_Entry ? EntryLoc : ExitLoc;
  }

private:
  CFGBlockInfo(LocalVarContext EmptyCtx)
      : EntryContext(EmptyCtx), ExitContext(EmptyCtx) {}

public:
  static CFGBlockInfo getEmptyBlockInfo(LocalVariableMap &M);
};

// A LocalVariableMap maintains a map from local variables to their currently
// valid definitions.  It provides SSA-like functionality when traversing the
// CFG.  Like SSA, each definition or assignment to a variable is assigned a
// unique name (an integer), which acts as the SSA name for that definition.
// The total set of names is shared among all CFG basic blocks.
// Unlike SSA, we do not rewrite expressions to replace local variables declrefs
// with their SSA-names.  Instead, we compute a Context for each point in the
// code, which maps local variables to the appropriate SSA-name.  This map
// changes with each assignment.
//
// The map is computed in a single pass over the CFG.  Subsequent analyses can
// then query the map to find the appropriate Context for a statement, and use
// that Context to look up the definitions of variables.
class LocalVariableMap {
public:
  using Context = LocalVarContext;

  /// A VarDefinition consists of an expression, representing the value of the
  /// variable, along with the context in which that expression should be
  /// interpreted.  A reference VarDefinition does not itself contain this
  /// information, but instead contains a pointer to a previous VarDefinition.
  struct VarDefinition {
  public:
    friend class LocalVariableMap;

    // The original declaration for this variable.
    const NamedDecl *Dec;

    // The expression for this variable, OR
    const Expr *Exp = nullptr;

    // Direct reference to another VarDefinition; for a merge ("phi"), the
    // definition on the first joined path.
    unsigned DirectRef = 0;

    // Reference to underlying canonical non-reference VarDefinition.
    unsigned CanonicalRef = 0;

    // For a merge ("phi") of two definitions, the definition on the second
    // joined path (DirectRef holds the first); 0 otherwise. A phi is its own
    // canonical definition and is opaque to lookupExpr(); it exists so that
    // a branch on a try-acquire result merged with a constant initializer
    // can still be resolved (see decodeTrylockCond()).
    unsigned PhiAlt = 0;

    // The map with which Exp should be interpreted.
    Context Ctx;

    // Whether this is the definition created at the variable's declaration
    // when it has no initializer: the variable's birth, holding an
    // indeterminate value. An invalidated reference has the same null
    // shape but stands for an unknown later value (chainAvoids()).
    bool UninitDecl = false;

    bool isPhi() const { return PhiAlt != 0; }
    bool isReference() const { return !Exp && !isPhi(); }

    void invalidateRef() { DirectRef = CanonicalRef = PhiAlt = 0; }

  private:
    // Create ordinary variable definition
    VarDefinition(const NamedDecl *D, const Expr *E, Context C)
        : Dec(D), Exp(E), Ctx(C), UninitDecl(!E) {}

    // Create reference to previous definition
    VarDefinition(const NamedDecl *D, unsigned DirectRef, unsigned CanonicalRef,
                  Context C)
        : Dec(D), DirectRef(DirectRef), CanonicalRef(CanonicalRef), Ctx(C) {}
  };

private:
  Context::Factory ContextFactory;
  std::vector<VarDefinition> VarDefinitions;
  std::vector<std::pair<const Stmt *, Context>> SavedContexts;
  // Whether the function contains a try-acquire call (a call or a
  // construction whose callee is annotated try_acquire_capability), found
  // by a scan of the CFG before the map is built (traverseCFG()). Only a
  // branch on a try-acquire's stored result ever reads a merged
  // definition ("phi"), a constant-folded join or an escape mark
  // (decodeTrylockCond()), so none of them is computed for a function
  // without one: its joins intersect by definition identity alone, as they
  // always did.
  bool TracksTryAcquires = false;
  // Whether anything reads a variable's definition back in this run: a
  // try-acquire's stored result is one reason (TracksTryAcquires), a beta
  // run's capability translation through a local (`p->mu` after `p = &f`)
  // the other. With neither, no definition is ever consulted, and keeping
  // the map exact across a mutation that carries no assignment is pure
  // cost (VarMapBuilder::VisitUnaryOperator()).
  bool ReadsDefinitions = false;
  // The function's try-acquire calls, each with the context after it, memoized
  // in traversal order. Constructions are not listed: they record in-walk,
  // where the constructed object's placeholder exists.
  SmallVector<std::pair<const CallExpr *, Context>, 4> TryAcquireCalls;
  // The calls whose result some definition stores in a local variable
  // (storesResultOf()), recorded as the definitions are added.
  llvm::SmallPtrSet<const CallExpr *, 4> StoredResults;
  // Variables whose storage is reachable through an escaped reference
  // (address taken, captured or bound by non-const reference), with the
  // blocks the escape happens in: a mutation through the reference is
  // invisible to the map, so from the escape onwards neither the
  // variable's definitions nor its merges identify its value. Where the
  // escape cannot reach, they still do -- an address taken after a checked
  // region must not defeat what the region proved (see decodeTrylockCond()).
  llvm::SmallDenseMap<const NamedDecl *, llvm::SmallVector<const CFGBlock *, 2>,
                      4>
      EscapedAt;
  // Where each escaped variable with automatic storage is declared: passing
  // the declaration again is fresh storage, which no earlier escape can
  // reach into (escapeReaches()).
  llvm::SmallDenseMap<const NamedDecl *, const CFGBlock *, 4> DeclaredIn;
  // Memoized constant values of canonical definitions, keyed by definition
  // ID (std::nullopt: does not constant-evaluate): intersectContexts()
  // consults the same definitions at every join they reach.
  llvm::DenseMap<unsigned, std::optional<llvm::APSInt>> ConstantValues;
  // Per variable, the definitions whose chain of prior definitions holds
  // constants only, all the way to the variable's declaration
  // (chainNonConstantDefs()). A chain passes only through its own
  // variable's definitions, and only intersectBackEdge() ever changes an
  // existing definition: it drops the variable's entry when it does.
  llvm::DenseMap<const NamedDecl *, llvm::SmallDenseSet<unsigned, 4>>
      CleanChains;
  // Memoized chainNonConstantDefs() results, keyed by the definition the
  // chain leads up to (std::nullopt: the chain reaches an unknown
  // definition, which is what the walk returns false for). The chain is a
  // pure function of the definition graph, and every join a variable
  // reaches asks about the same few definitions, so without this a
  // variable with one non-constant definition makes the joins quadratic in
  // their own number. Invalidated with CleanChains, by the one thing that
  // changes an existing definition (intersectBackEdge()).
  llvm::DenseMap<unsigned, std::optional<llvm::SmallDenseSet<unsigned, 8>>>
      ChainDefs;

public:
  LocalVariableMap() {
    // index 0 is a placeholder for undefined variables (aka phi-nodes).
    VarDefinitions.push_back(VarDefinition(nullptr, 0, 0, getEmptyContext()));
  }

  /// Look up a definition, within the given context.
  const VarDefinition* lookup(const NamedDecl *D, Context Ctx) {
    const unsigned *i = Ctx.lookup(D);
    if (!i)
      return nullptr;
    assert(*i < VarDefinitions.size());
    return &VarDefinitions[*i];
  }

  /// Look up the canonical definition for \p D within the given context:
  /// the definition its reference chain resolves to (e.g. a loop head wraps
  /// every incoming definition in a reference).  Returns NULL if the
  /// variable is not in the context or resolves to an unknown definition.
  const VarDefinition *lookupCanonical(const NamedDecl *D, Context Ctx) {
    const unsigned *i = Ctx.lookup(D);
    if (!i)
      return nullptr;
    assert(*i < VarDefinitions.size());
    unsigned ID = getCanonicalDefinitionID(*i);
    return ID ? &VarDefinitions[ID] : nullptr;
  }

  /// Look up the expression for the definition \p i, looking through
  /// references. Returns NULL if the expression is not statically known --
  /// including for a phi, which has no single defining expression. If
  /// successful, also modifies Ctx to hold the context of the returned Expr.
  const Expr *lookupExprByID(unsigned i, Context &Ctx) {
    while (i > 0) {
      const VarDefinition &VD = VarDefinitions[i];
      if (VD.Exp) {
        Ctx = VD.Ctx;
        return VD.Exp;
      }
      if (VD.isPhi())
        return nullptr;
      i = VD.DirectRef;
    }
    return nullptr;
  }

  /// Look up the definition for D within the given context.  Returns
  /// NULL if the expression is not statically known.  If successful, also
  /// modifies Ctx to hold the context of the return Expr.
  const Expr* lookupExpr(const NamedDecl *D, Context &Ctx) {
    const unsigned *P = Ctx.lookup(D);
    if (!P)
      return nullptr;
    return lookupExprByID(*P, Ctx);
  }

  void markEscaped(const NamedDecl *D, const CFGBlock *B) {
    auto &Blocks = EscapedAt[D];
    if (Blocks.empty() || Blocks.back() != B)
      Blocks.push_back(B);
  }
  /// The blocks in which \p D's storage escapes, empty if it never does.
  ArrayRef<const CFGBlock *> escapedAt(const NamedDecl *D) const {
    auto It = EscapedAt.find(D);
    return It == EscapedAt.end() ? ArrayRef<const CFGBlock *>()
                                 : ArrayRef<const CFGBlock *>(It->second);
  }
  void markDeclaredIn(const NamedDecl *D, const CFGBlock *B) {
    DeclaredIn.try_emplace(D, B);
  }
  /// The block declaring \p D's automatic storage, or null.
  const CFGBlock *declaredIn(const NamedDecl *D) const {
    auto It = DeclaredIn.find(D);
    return It == DeclaredIn.end() ? nullptr : It->second;
  }

  /// What walkChain() does with a definition it has just visited.
  enum class ChainVisit {
    Fail,   ///< The walk's question is answered: stop and return false.
    Follow, ///< Continue into this definition's own prior definitions.
    Prune,  ///< This definition's chain adds nothing: do not follow it.
  };

  /// Walks the chain of \p D's definitions leading up to definition \p ID:
  /// every path of prior definitions from \p ID back to \p D's declaration,
  /// calling \p Visit on each definition passed through (a merge continues
  /// into both of its operands' chains). Returns false if \p Visit rejects
  /// a definition, or if a path reaches an unknown definition, about which
  /// nothing can be concluded; true if every path ended at the declaration.
  /// A loop back edge passes the loop-head definition as \p StopAt: a chain
  /// that reaches the head has been walked as far as the iteration goes,
  /// and what precedes the head is the merge being tested itself.
  template <typename VisitFn>
  bool walkChain(const NamedDecl *D, unsigned ID, unsigned StopAt,
                 VisitFn Visit) {
    SmallVector<unsigned, 4> Worklist = {ID};
    llvm::SmallDenseSet<unsigned, 8> Visited;
    while (!Worklist.empty()) {
      unsigned ID = Worklist.pop_back_val();
      // Resolve references one step at a time: \p StopAt is usually a
      // loop-head reference, which one-hop canonicalization would skip
      // right past.
      bool PathEnds = false;
      while (ID > 0 && VarDefinitions[ID].isReference()) {
        if (ID == StopAt || VarDefinitions[ID].UninitDecl) {
          // The loop head ends the path (see above); so does the variable's
          // declaration, with or without an initializer.
          PathEnds = true;
          break;
        }
        ID = VarDefinitions[ID].DirectRef;
      }
      if (PathEnds || (StopAt != 0 && ID == StopAt))
        continue; // A phi-converted loop head is its own canonical.
      if (ID == 0)
        return false;
      if (!Visited.insert(ID).second)
        continue; // A phi-converted loop head can make the graph cyclic.
      ChainVisit Step = Visit(ID);
      if (Step == ChainVisit::Fail)
        return false;
      if (Step == ChainVisit::Prune)
        continue;
      if (VarDefinitions[ID].isPhi()) {
        // The merged value is one of the operands': the chain continues
        // into both.
        Worklist.push_back(VarDefinitions[ID].DirectRef);
        Worklist.push_back(VarDefinitions[ID].PhiAlt);
        continue;
      }
      const unsigned *P = VarDefinitions[ID].Ctx.lookup(D);
      if (P)
        Worklist.push_back(*P);
      // Otherwise this path reached the declaration.
    }
    return true;
  }

  /// Returns true if the chain of \p D's definitions leading up to
  /// definition \p ID provably does not contain definition \p Avoid: every
  /// path of prior definitions from \p ID reaches \p D's declaration
  /// without passing \p Avoid or an unknown definition (walkChain()). Used
  /// to establish that the assignment creating \p Avoid was never executed
  /// on the paths where \p ID is the reaching definition. \p StopAt is as
  /// in walkChain(): a chain that reaches the loop head avoided \p Avoid
  /// within the iteration.
  bool chainAvoids(const NamedDecl *D, unsigned ID, unsigned Avoid,
                   unsigned StopAt = 0) {
    Avoid = getCanonicalDefinitionID(Avoid);
    return walkChain(D, ID, StopAt, [Avoid](unsigned Def) {
      return Def == Avoid ? ChainVisit::Fail : ChainVisit::Follow;
    });
  }

  /// Collects into \p Defs every non-constant definition (a merge included)
  /// that the chain of \p D's definitions leading up to \p ID passes
  /// through: the definitions a chainAvoids() query on \p ID answers "no"
  /// for, provided the definition asked about is itself non-constant. That
  /// is what resolution asks about -- decodeTrylockCond() and phiAbsorbs()
  /// both name a merge's non-constant operand -- but not what every caller
  /// asks: the back-edge unwrap in intersectBackEdge() can name a constant
  /// operand, and \p Defs deliberately says nothing about those
  /// (constantToKeep() is not consulted there). Returns false if the chain
  /// reaches an unknown definition, in which case \p Defs says nothing at
  /// all.
  ///
  /// A chain that contributes nothing -- constant definitions all the way
  /// to the declaration -- is memoized (CleanChains) and pruned when a
  /// later walk reaches it, so that a variable assigned constants over and
  /// over does not make every join walk its whole history.
  bool chainNonConstantDefs(const NamedDecl *D, unsigned ID,
                            llvm::SmallDenseSet<unsigned, 8> &Defs) {
    if (auto It = ChainDefs.find(ID); It != ChainDefs.end()) {
      if (!It->second)
        return false;
      Defs.insert(It->second->begin(), It->second->end());
      return true;
    }
    const auto CleanIt = CleanChains.find(D);
    llvm::SmallDenseSet<unsigned, 8> Walked;
    bool Known = walkChain(D, ID, /*StopAt=*/0, [&](unsigned Def) {
      // A definition whose own chain has been walked before answers for
      // everything behind it: that is what makes a run of joins linear
      // rather than quadratic, since each join's walk stops at the
      // definition the previous join left behind.
      if (auto It = ChainDefs.find(Def); It != ChainDefs.end()) {
        if (!It->second)
          return ChainVisit::Fail;
        Walked.insert(It->second->begin(), It->second->end());
        return ChainVisit::Prune;
      }
      if (CleanIt != CleanChains.end() && CleanIt->second.contains(Def))
        return ChainVisit::Prune;
      if (!constantValue(Def))
        Walked.insert(Def);
      return ChainVisit::Follow;
    });
    if (Known && Walked.empty())
      if (unsigned Canon = getCanonicalDefinitionID(ID))
        CleanChains[D].insert(Canon);
    if (Known) {
      Defs.insert(Walked.begin(), Walked.end());
      ChainDefs[ID] = std::move(Walked);
    } else {
      ChainDefs[ID] = std::nullopt;
    }
    return Known;
  }

  /// The constant integer value of the canonical definition \p Canon,
  /// memoized; std::nullopt if the definition is unknown, a merge, or does
  /// not constant-evaluate. Any expression that constant-evaluates counts,
  /// not just a literal: `bool b = kFalseConstant;` is the constant false.
  std::optional<llvm::APSInt> constantValue(unsigned Canon) {
    if (Canon == 0 || VarDefinitions[Canon].isPhi())
      return std::nullopt;
    auto [It, Inserted] = ConstantValues.try_emplace(Canon);
    if (Inserted) {
      const Expr *E = VarDefinitions[Canon].Exp;
      Expr::EvalResult ER;
      if (E && !E->isValueDependent() &&
          E->EvaluateAsInt(ER, VarDefinitions[Canon].Dec->getASTContext()))
        It->second = ER.Val.getInt();
    }
    return It->second;
  }

  /// Whether the canonical definitions \p Canon1 and \p Canon2 constant-
  /// evaluate to the same integer value: e.g. after
  /// `bool b = false; if (c) b = false;` the variable is still the constant
  /// false, and can later merge with a try-acquire result (a merge of
  /// merges is not resolved). The values must match exactly, not merely in
  /// truthiness, and non-integer constants (e.g. two distinct addresses,
  /// which are both "true") never match. Shared by the branch-join and
  /// back-edge merge engines (intersectContexts() / intersectBackEdge())
  /// so that they cannot drift apart.
  bool valueEqualConstants(unsigned Canon1, unsigned Canon2) {
    std::optional<llvm::APSInt> V1 = constantValue(Canon1);
    if (!V1)
      return false;
    std::optional<llvm::APSInt> V2 = constantValue(Canon2);
    return V2 && llvm::APSInt::isSameValue(*V1, *V2);
  }

  /// Which of two value-equal constant definitions of \p D a branch join
  /// may keep for the other, or 0 if neither will do. Equal values alone
  /// do not make the two interchangeable: resolving a merge asks whether
  /// the constant's chain passes the try-acquire call (chainAvoids()), and
  /// the definition that is kept answers that for both paths. In
  /// `if (c1) { ok = mu.TryLock(); if (c2) ok = false; } else ok = false;`
  /// only the first `ok = false` can follow the call, so keeping the else
  /// arm's in its place would resolve a merge that must not resolve.
  ///
  /// The merged value's chain is really the union of the two, so the one
  /// to keep is the one whose non-constant definitions
  /// (chainNonConstantDefs(), the definitions resolution's queries name)
  /// already cover the other's -- it then answers every such query exactly
  /// as the union would, not merely conservatively. A chain that reaches an
  /// unknown definition covers everything, being answered "no" throughout.
  /// When neither covers the other, the join keeps no constant and merges
  /// them.
  unsigned constantToKeep(const NamedDecl *D, unsigned Canon1,
                          unsigned Canon2) {
    if (!valueEqualConstants(Canon1, Canon2))
      return 0;
    llvm::SmallDenseSet<unsigned, 8> Defs1, Defs2;
    if (!chainNonConstantDefs(D, Canon1, Defs1))
      return Canon1;
    if (!chainNonConstantDefs(D, Canon2, Defs2))
      return Canon2;
    if (llvm::set_is_subset(Defs2, Defs1))
      return Canon1;
    if (llvm::set_is_subset(Defs1, Defs2))
      return Canon2;
    return 0;
  }

  /// Whether the merge \p CanonPhi already covers the definition
  /// \p CanonOther of variable \p Dec, so that joining the two keeps the
  /// phi as is: \p CanonOther is one of the phi's own operands, or a
  /// definition that is not an operand but is value-equal to the phi's
  /// constant operand (constants of the same value are interchangeable,
  /// valueEqualConstants()) -- provided its chain avoids the phi's
  /// non-constant operand, exactly as resolving the phi imposes on the
  /// recorded constant (chainAvoids()): e.g. phi(call, false) absorbs
  /// another `= false` assignment that cannot follow the call. \p StopAt
  /// is forwarded to chainAvoids() by loop back edges. Like
  /// valueEqualConstants(), shared by both merge engines.
  bool phiAbsorbs(const NamedDecl *Dec, unsigned CanonPhi, unsigned CanonOther,
                  unsigned StopAt = 0) {
    if (CanonPhi == 0 || CanonOther == 0 || !VarDefinitions[CanonPhi].isPhi())
      return false;
    const VarDefinition &VD = VarDefinitions[CanonPhi];
    unsigned Op1 = getCanonicalDefinitionID(VD.DirectRef);
    unsigned Op2 = getCanonicalDefinitionID(VD.PhiAlt);
    if (Op1 == CanonOther || Op2 == CanonOther)
      return true;
    std::optional<llvm::APSInt> VO = constantValue(CanonOther);
    if (!VO)
      return false;
    std::optional<llvm::APSInt> V1 = constantValue(Op1);
    std::optional<llvm::APSInt> V2 = constantValue(Op2);
    if (V1 && V2)
      // Both operands constant: absorb a matching value. Unlike a join of
      // two constants (constantToKeep()), this keeps the phi without asking
      // whether the absorbed definition's chain is covered by the operands'
      // -- a phi of two constants has no non-constant operand, so it
      // resolves no branch by itself, and a call the absorbed chain passed
      // was overwritten before this join, which leaves its try fact
      // unchecked for the lockset join to report.
      return llvm::APSInt::isSameValue(*V1, *VO) ||
             llvm::APSInt::isSameValue(*V2, *VO);
    std::optional<llvm::APSInt> VC = V1 ? V1 : V2;
    unsigned NonConstOp = V1 ? Op2 : Op1;
    return VC && llvm::APSInt::isSameValue(*VC, *VO) &&
           chainAvoids(Dec, CanonOther, NonConstOp, StopAt);
  }

  Context getEmptyContext() { return ContextFactory.getEmptyMap(); }

  /// Return the next context after processing S.  This function is used by
  /// clients of the class to get the appropriate context when traversing the
  /// CFG.  It must be called for every assignment or DeclStmt.
  const Context &getNextContext(unsigned &CtxIndex, const Stmt *S,
                                const Context &C) {
    if (SavedContexts[CtxIndex + 1].first == S) {
      CtxIndex++;
      const Context &Result = SavedContexts[CtxIndex].second;
      return Result;
    }
    return C;
  }

  void dumpVarDefinitionName(unsigned i) {
    if (i == 0) {
      llvm::errs() << "Undefined";
      return;
    }
    const NamedDecl *Dec = VarDefinitions[i].Dec;
    if (!Dec) {
      llvm::errs() << "<<NULL>>";
      return;
    }
    Dec->printName(llvm::errs());
    llvm::errs() << "." << i << " " << ((const void*) Dec);
  }

  /// Dumps an ASCII representation of the variable map to llvm::errs()
  void dump() {
    for (unsigned i = 1, e = VarDefinitions.size(); i < e; ++i) {
      const Expr *Exp = VarDefinitions[i].Exp;
      unsigned Ref = VarDefinitions[i].DirectRef;

      dumpVarDefinitionName(i);
      llvm::errs() << " = ";
      if (Exp) Exp->dump();
      else {
        dumpVarDefinitionName(Ref);
        llvm::errs() << "\n";
      }
    }
  }

  /// Dumps an ASCII representation of a Context to llvm::errs()
  void dumpContext(Context C) {
    for (Context::iterator I = C.begin(), E = C.end(); I != E; ++I) {
      const NamedDecl *D = I.getKey();
      D->printName(llvm::errs());
      llvm::errs() << " -> ";
      dumpVarDefinitionName(I.getData());
      llvm::errs() << "\n";
    }
  }

  /// Builds the variable map.
  void traverseCFG(AnalysisDeclContext &AC, CFG *CFGraph,
                   const PostOrderCFGView *SortedGraph,
                   std::vector<CFGBlockInfo> &BlockInfo, bool BetaWarnings);

  /// Whether the function contains any try-acquire call (see
  /// TracksTryAcquires).
  bool tracksTryAcquires() const { return TracksTryAcquires; }
  /// Whether any consumer reads variable definitions back (see
  /// ReadsDefinitions).
  bool readsDefinitions() const { return ReadsDefinitions; }
  /// The function's try-acquire calls, each with the context after it.
  ArrayRef<std::pair<const CallExpr *, Context>> tryAcquireCalls() const {
    return TryAcquireCalls;
  }

  /// Whether the result of \p Call is stored in a local variable anywhere in
  /// the function. Only a stored result can be branched on away from the
  /// call itself: a condition naming a variable is what the terminator
  /// decode resolves back to the call (decodeTrylockCond()), so a result
  /// that is never stored is branched on by the terminator containing the
  /// call and by no other.
  bool storesResultOf(const CallExpr *Call) const {
    return StoredResults.contains(Call);
  }

protected:
  friend class VarMapBuilder;

  // Resolve any definition ID down to its non-reference base ID.
  //
  // This follows the CanonicalRef each reference caches when it is created
  // (addReference()), which intersectBackEdge() can outdate: it converts a
  // loop-head reference into a phi, or invalidates it, in place -- after an
  // inner loop's head has wrapped that reference in one of its own, and
  // cached the base it resolved to back then. Stepping through DirectRef
  // instead reaches the mutated head, so the two walks can disagree for a
  // reference created inside a loop whose head is merged later. Nothing
  // depends on the difference today: the consumers of this cache either run
  // before the mutation or re-check isPhi() on what they get back, and a
  // walk that must see the current state resolves references one at a time
  // (walkChain()). A new consumer must not assume otherwise.
  unsigned getCanonicalDefinitionID(unsigned ID) const {
    while (ID > 0 && VarDefinitions[ID].isReference())
      ID = VarDefinitions[ID].CanonicalRef;
    return ID;
  }

  // Get the current context index
  unsigned getContextIndex() { return SavedContexts.size()-1; }

  // Note a call whose result a definition stores (storesResultOf()).
  void recordStoredResult(const Expr *Exp) {
    if (const auto *Call = dyn_cast_or_null<CallExpr>(
            Exp ? Exp->IgnoreParenImpCasts() : nullptr))
      StoredResults.insert(Call);
  }

  // Save the current context for later replay
  void saveContext(const Stmt *S, Context C) {
    SavedContexts.push_back(std::make_pair(S, C));
  }

  // Adds a new definition to the given context, and returns a new context.
  // This method should be called when declaring a new variable.
  Context addDefinition(const NamedDecl *D, const Expr *Exp, Context Ctx) {
    assert(!Ctx.contains(D));
    recordStoredResult(Exp);
    unsigned newID = VarDefinitions.size();
    Context NewCtx = ContextFactory.add(Ctx, D, newID);
    VarDefinitions.push_back(VarDefinition(D, Exp, Ctx));
    return NewCtx;
  }

  // Add a new reference to an existing definition.
  Context addReference(const NamedDecl *D, unsigned Ref, Context Ctx) {
    unsigned newID = VarDefinitions.size();
    Context NewCtx = ContextFactory.add(Ctx, D, newID);
    VarDefinitions.push_back(
        VarDefinition(D, Ref, getCanonicalDefinitionID(Ref), Ctx));
    return NewCtx;
  }

  // Merge two distinct definitions into a phi definition: the variable's
  // value is that of one of the two. Most consumers treat a phi like a
  // cleared definition; see VarDefinition::PhiAlt for why it exists.
  Context addPhiDefinition(const NamedDecl *D, unsigned Ref1, unsigned Ref2,
                           Context Ctx) {
    assert(Ref1 && Ref2 && "phi operands must be known definitions");
    unsigned newID = VarDefinitions.size();
    Context NewCtx =
        ContextFactory.add(ContextFactory.remove(Ctx, D), D, newID);
    VarDefinition VD(D, Ref1, /*CanonicalRef=*/0, Ctx);
    VD.PhiAlt = Ref2;
    VarDefinitions.push_back(VD);
    return NewCtx;
  }

  // Updates a definition only if that definition is already in the map.
  // This method should be called when assigning to an existing variable.
  Context updateDefinition(const NamedDecl *D, Expr *Exp, Context Ctx) {
    recordStoredResult(Exp);
    if (Ctx.contains(D)) {
      unsigned newID = VarDefinitions.size();
      Context NewCtx = ContextFactory.remove(Ctx, D);
      NewCtx = ContextFactory.add(NewCtx, D, newID);
      VarDefinitions.push_back(VarDefinition(D, Exp, Ctx));
      return NewCtx;
    }
    return Ctx;
  }

  // Removes a definition from the context, but keeps the variable name
  // as a valid variable.  The index 0 is a placeholder for cleared definitions.
  Context clearDefinition(const NamedDecl *D, Context Ctx) {
    Context NewCtx = Ctx;
    if (NewCtx.contains(D)) {
      NewCtx = ContextFactory.remove(NewCtx, D);
      NewCtx = ContextFactory.add(NewCtx, D, 0);
    }
    return NewCtx;
  }

  // Remove a definition entirely frmo the context.
  Context removeDefinition(const NamedDecl *D, Context Ctx) {
    Context NewCtx = Ctx;
    if (NewCtx.contains(D)) {
      NewCtx = ContextFactory.remove(NewCtx, D);
    }
    return NewCtx;
  }

  Context intersectContexts(Context C1, Context C2);
  Context createReferenceContext(Context C);
  void intersectBackEdge(Context C1, Context C2);
};

} // namespace

// This has to be defined after LocalVariableMap.
CFGBlockInfo CFGBlockInfo::getEmptyBlockInfo(LocalVariableMap &M) {
  return CFGBlockInfo(M.getEmptyContext());
}

namespace {

/// Visitor which builds a LocalVariableMap
class VarMapBuilder : public ConstStmtVisitor<VarMapBuilder> {
public:
  /// The block whose statements are being visited: where an escape found
  /// here happens (LocalVariableMap::markEscaped()).
  const CFGBlock *CurBlock = nullptr;

  LocalVariableMap* VMap;
  LocalVariableMap::Context Ctx;

  VarMapBuilder(LocalVariableMap *VM, LocalVariableMap::Context C,
                AnalysisDeclContext &AC)
      : VMap(VM), Ctx(C), AC(AC) {}

  void VisitDeclStmt(const DeclStmt *S);
  void VisitBinaryOperator(const BinaryOperator *BO);
  void VisitUnaryOperator(const UnaryOperator *UO);
  void VisitLambdaExpr(const LambdaExpr *LE);
  void VisitBlockExpr(const BlockExpr *BE);
  void VisitInitListExpr(const InitListExpr *ILE);
  void VisitCallExpr(const CallExpr *CE);
  void VisitCXXConstructExpr(const CXXConstructExpr *CE);

private:
  // Only used to reach the body's parent map, and only for an address-of
  // expression: the map is built lazily, so functions that take no address
  // never pay for it.
  AnalysisDeclContext &AC;

  void markEscapedIfDeclRef(const Expr *E);
  void markEscapedRefBindings(const InitListExpr *ILE);
  void clearCallMutations(const CallExpr *CE);
};

} // namespace

// The one rule for marking a variable whose storage becomes reachable
// through a reference: it can then be mutated without a visible assignment.
// Shared by every escape site so they cannot drift apart; IgnoreParenCasts,
// because an explicit cast (`(bool &)b`) hides the variable just as well as
// an implicit one.
void VarMapBuilder::markEscapedIfDeclRef(const Expr *E) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenCasts()))
    VMap->markEscaped(DRE->getDecl(), CurBlock);
}

// Marks variables bound to non-const reference members in an aggregate
// initialization (`struct W { bool &b; }; W w{ok};`): the aggregate can
// mutate them without a visible assignment, like any reference binding.
void VarMapBuilder::markEscapedRefBindings(const InitListExpr *ILE) {
  // Descends into an initializer that is itself a list (an array element, a
  // base class, a nested aggregate member).
  auto Descend = [this](const Expr *Init) {
    if (const auto *Nested = dyn_cast_or_null<InitListExpr>(
            Init ? Init->IgnoreParenImpCasts() : nullptr))
      markEscapedRefBindings(Nested);
  };

  // An array of aggregates has no fields of its own: every element carries
  // its own initializer list.
  if (ILE->getType()->isArrayType()) {
    for (const Expr *Init : ILE->inits())
      Descend(Init);
    Descend(ILE->getArrayFiller());
    return;
  }

  const RecordDecl *RD = ILE->getType()->getAsRecordDecl();
  if (!RD || RD->isUnion())
    return; // A union cannot have a reference member.
  unsigned I = 0, N = ILE->getNumInits();
  // C++17 aggregate initialization lists the base classes before the
  // fields, so the field walk starts past them or the two run out of step.
  if (const auto *CXXRD = dyn_cast<CXXRecordDecl>(RD))
    for (unsigned NumBases = CXXRD->getNumBases(); I < N && I < NumBases; ++I)
      Descend(ILE->getInit(I));
  auto FI = RD->field_begin(), FE = RD->field_end();
  for (; I < N && FI != FE; ++I, ++FI) {
    // An unnamed bit-field takes no initializer of its own; stepping over
    // it with the initializer index would pair every later field with the
    // wrong one.
    while (FI != FE && FI->isUnnamedBitField())
      ++FI;
    if (FI == FE)
      break;
    const Expr *Init = ILE->getInit(I);
    if (!Init)
      continue;
    QualType FT = FI->getType();
    if (FT->isReferenceType() && !FT.getNonReferenceType().isConstQualified())
      markEscapedIfDeclRef(Init);
    else
      Descend(Init);
  }
}

// Add new local variables to the variable map
void VarMapBuilder::VisitDeclStmt(const DeclStmt *S) {
  bool modifiedCtx = false;
  const DeclGroupRef DGrp = S->getDeclGroup();
  for (const auto *D : DGrp) {
    if (const auto *VD = dyn_cast_or_null<VarDecl>(D)) {
      const Expr *E = VD->getInit();

      // Add local variables with trivial type to the variable map
      QualType T = VD->getType();
      if (T.isTrivialType(VD->getASTContext())) {
        Ctx = VMap->addDefinition(VD, E, Ctx);
        if (VD->hasLocalStorage() && !VD->isStaticLocal())
          VMap->markDeclaredIn(VD, CurBlock);
        modifiedCtx = true;
      } else if (T->isReferenceType() && E &&
                 !T.getNonReferenceType().isConstQualified()) {
        // Binding a non-const reference to a variable lets the variable be
        // mutated without a visible assignment.
        markEscapedIfDeclRef(E);
      }
      // Aggregate initialization can bind non-const reference members.
      if (const auto *ILE = dyn_cast_or_null<InitListExpr>(
              E ? E->IgnoreParenImpCasts() : nullptr))
        markEscapedRefBindings(ILE);
    }
  }
  if (modifiedCtx)
    VMap->saveContext(S, Ctx);
}

// Update local variable definitions in variable map
void VarMapBuilder::VisitBinaryOperator(const BinaryOperator *BO) {
  if (!BO->isAssignmentOp())
    return;

  Expr *LHSExp = BO->getLHS()->IgnoreParenCasts();

  // Update the variable map and current context.
  if (const auto *DRE = dyn_cast<DeclRefExpr>(LHSExp)) {
    const ValueDecl *VDec = DRE->getDecl();
    if (Ctx.lookup(VDec)) {
      if (BO->getOpcode() == BO_Assign)
        Ctx = VMap->updateDefinition(VDec, BO->getRHS(), Ctx);
      else
        // FIXME -- handle compound assignment operators
        Ctx = VMap->clearDefinition(VDec, Ctx);
      VMap->saveContext(BO, Ctx);
    }
  }
}

// True if the address the expression produces can only be read through:
// it is handed straight to a call as a pointer-to-const argument, and
// nothing else. `observe(&b)` with `void observe(const bool *)` is the
// shape this recognizes -- without it, passing `&b` to a const-taking API
// would lose the plain `bool b = mu.TryLock(); if (b) ...` form.
//
// The address is followed out through the conversions it flows into, to
// the last one that is still a pointer, and only that outermost type
// decides: an intermediate `const bool *` proves nothing when a cast
// strips the const again (`const_cast<bool *>(static_cast<const bool *>
// (&b))`). It must also end there, as an argument of the call: a
// pointer-to-const that is *stored* proves nothing either, since the
// const can be cast away anywhere the stored pointer reaches
// (`const bool *cp = &b; *const_cast<bool *>(cp) = true;`).
static bool addrOfIsReadOnly(const UnaryOperator *UO, ParentMap &PM) {
  const Expr *E = UO;
  while (true) {
    // Only the conversions the address itself flows through are followed;
    // any other parent consumes the pointer as it stands.
    const Stmt *P = PM.getParent(E);
    if (!P || !(isa<ParenExpr>(P) || isa<CastExpr>(P)))
      break;
    // Once it is no longer a pointer (`(void)&b`, a cast to an integer)
    // the address cannot be followed any further; the last pointer type it
    // had is the one that reached that conversion.
    const auto *PE = cast<Expr>(P);
    if (!PE->getType()->isPointerType())
      break;
    E = PE;
  }
  QualType T = E->getType();
  if (!T->isPointerType() || !T->getPointeeType().isConstQualified())
    return false;
  const Stmt *Consumer = PM.getParent(E);
  return isa_and_nonnull<CallExpr>(Consumer) ||
         isa_and_nonnull<CXXConstructExpr>(Consumer);
}

// An increment or decrement mutates the variable like a compound assignment
// (VisitBinaryOperator()): its definition no longer identifies the stored
// value. Otherwise, marks a variable whose address is taken: it can then be
// mutated without a visible assignment. An address that is only readable
// through is not an escape, the same const distinction VisitDeclStmt(),
// VisitCallExpr() and VisitCXXConstructExpr() make for reference and pointer
// parameters -- without it, passing `&b` to a const-taking API would lose
// the plain `bool b = mu.TryLock(); if (b) ...` form.
void VarMapBuilder::VisitUnaryOperator(const UnaryOperator *UO) {
  if (UO->isIncrementDecrementOp()) {
    // Gated like the address-of branch below: with no consumer for the
    // definitions, clearing one costs a VarDefinition and a rebuilt
    // ImmutableMap for nothing, on every increment in the translation unit.
    if (!VMap->readsDefinitions())
      return;
    if (const auto *DRE =
            dyn_cast<DeclRefExpr>(UO->getSubExpr()->IgnoreParenCasts())) {
      const ValueDecl *VDec = DRE->getDecl();
      if (Ctx.lookup(VDec)) {
        Ctx = VMap->clearDefinition(VDec, Ctx);
        VMap->saveContext(UO, Ctx);
      }
    }
    return;
  }
  if (UO->getOpcode() != UO_AddrOf || !VMap->tracksTryAcquires())
    return;
  // Checked before the parent map is touched: building it is what makes
  // this more than a type test, and nothing else here needs it.
  if (!isa<DeclRefExpr>(UO->getSubExpr()->IgnoreParenCasts()))
    return;
  if (addrOfIsReadOnly(UO, AC.getParentMap()))
    return;
  markEscapedIfDeclRef(UO->getSubExpr());
}

// An initializer list can bind non-const reference members wherever it
// appears, not only as a declaration's initializer: `sink(W{ok})` and
// `new W{ok}` never reach VisitDeclStmt(), and `W w = W{ok}` reaches it
// behind a functional cast that IgnoreParenImpCasts() does not strip.
void VarMapBuilder::VisitInitListExpr(const InitListExpr *ILE) {
  markEscapedRefBindings(ILE);
}

// Marks variables a block literal captures by reference (`__block`): any
// later call of the block may mutate them without a visible assignment,
// exactly as a lambda's by-reference capture may.
void VarMapBuilder::VisitBlockExpr(const BlockExpr *BE) {
  for (const BlockDecl::Capture &C : BE->getBlockDecl()->captures())
    if (C.isByRef())
      VMap->markEscaped(C.getVariable(), CurBlock);
}

// Marks variables captured by reference in a lambda: any later call may
// mutate them without a visible assignment.
void VarMapBuilder::VisitLambdaExpr(const LambdaExpr *LE) {
  for (const LambdaCapture &LC : LE->captures()) {
    if (!LC.capturesVariable() || LC.getCaptureKind() != LCK_ByRef)
      continue;
    const ValueDecl *VD = LC.getCapturedVar();
    VMap->markEscaped(VD, CurBlock);
    // A reference init-capture (`[&x = b]`) binds like a reference
    // declaration: the escaped variable is the one in the initializer.
    if (const auto *IC = dyn_cast<VarDecl>(VD); IC && IC->isInitCapture())
      if (const Expr *Init = IC->getInit())
        markEscapedIfDeclRef(Init);
  }
}

// The context after a call: its non-const reference and pointer arguments'
// definitions cleared; then try-acquire capabilities added back.
void VarMapBuilder::VisitCallExpr(const CallExpr *CE) {
  clearCallMutations(CE);
  if (VMap->tracksTryAcquires())
    if (const auto *D = dyn_cast_or_null<NamedDecl>(CE->getCalleeDecl());
        D && D->hasAttr<TryAcquireCapabilityAttr>())
      VMap->TryAcquireCalls.emplace_back(CE, Ctx);
}

// Invalidates local variable definitions if variable escaped.
void VarMapBuilder::clearCallMutations(const CallExpr *CE) {
  const FunctionDecl *FD = CE->getDirectCallee();
  if (!FD)
    return;

  // Heuristic for likely-benign functions that pass by mutable reference. This
  // is needed to avoid a slew of false positives due to mutable reference
  // passing where the captured reference is usually passed on by-value.
  if (const IdentifierInfo *II = FD->getIdentifier()) {
    // Any kind of std::bind-like functions.
    if (II->isStr("bind") || II->isStr("bind_front"))
      return;
  }

  // Invalidate local variable definitions that are passed by non-const
  // reference or non-const pointer.
  for (unsigned Idx = 0; Idx < CE->getNumArgs(); ++Idx) {
    if (Idx >= FD->getNumParams())
      break;

    const Expr *Arg = CE->getArg(Idx)->IgnoreParenImpCasts();
    const ParmVarDecl *PVD = FD->getParamDecl(Idx);
    QualType ParamType = PVD->getType();

    // Potential reassignment if passed by non-const reference / pointer.
    const ValueDecl *VDec = nullptr;
    if (ParamType->isReferenceType() &&
        !ParamType->getPointeeType().isConstQualified()) {
      if (const auto *DRE = dyn_cast<DeclRefExpr>(Arg))
        VDec = DRE->getDecl();
    } else if (ParamType->isPointerType() &&
               !ParamType->getPointeeType().isConstQualified()) {
      Arg = Arg->IgnoreParenCasts();
      if (const auto *UO = dyn_cast<UnaryOperator>(Arg)) {
        if (UO->getOpcode() == UO_AddrOf) {
          const Expr *SubE = UO->getSubExpr()->IgnoreParenCasts();
          if (const auto *DRE = dyn_cast<DeclRefExpr>(SubE))
            VDec = DRE->getDecl();
        }
      }
    }

    if (VDec)
      Ctx = VMap->clearDefinition(VDec, Ctx);
  }
  // Save the context after the call where escaped variables' definitions (if
  // they exist) are cleared.
  VMap->saveContext(CE, Ctx);
}

// Marks variables bound to a constructor's non-const reference parameters:
// the constructed object can store the reference and mutate them later
// without a visible assignment. (VisitCallExpr() above only clears the
// definition for an ordinary call, which is assumed to mutate during the
// call but not to retain the reference.)
void VarMapBuilder::VisitCXXConstructExpr(const CXXConstructExpr *CE) {
  const CXXConstructorDecl *CD = CE->getConstructor();
  if (!CD)
    return;
  for (unsigned Idx = 0, N = CE->getNumArgs(); Idx < N; ++Idx) {
    if (Idx >= CD->getNumParams())
      break;
    QualType ParamType = CD->getParamDecl(Idx)->getType();
    if (ParamType->isReferenceType() &&
        !ParamType->getPointeeType().isConstQualified())
      markEscapedIfDeclRef(CE->getArg(Idx));
  }
}

// Computes the intersection of two contexts.  The intersection is the
// set of variables which have the same definition in both contexts;
// variables with different definitions are discarded.
LocalVariableMap::Context
LocalVariableMap::intersectContexts(Context C1, Context C2) {
  Context Result = C1;
  for (const auto &P : C1) {
    const NamedDecl *Dec = P.first;
    const unsigned *I2 = C2.lookup(Dec);
    if (!I2) {
      // The variable doesn't exist on second path.
      Result = removeDefinition(Dec, Result);
    } else if (P.second != *I2) {
      unsigned Canon1 = getCanonicalDefinitionID(P.second);
      unsigned Canon2 = getCanonicalDefinitionID(*I2);
      if (Canon1 == Canon2 && Canon1 != 0)
        continue; // Same underlying definition on both paths.
      // Nothing below is ever read in a function without a try-acquire
      // (see TracksTryAcquires): the definitions differ, and that is all.
      if (!TracksTryAcquires) {
        Result = clearDefinition(Dec, Result);
        continue;
      }
      // Distinct definitions that constant-evaluate to the same integer
      // value are interchangeable for resolution purposes, provided the
      // one kept answers every later chain query for both paths
      // (constantToKeep()).
      if (unsigned Keep = constantToKeep(Dec, Canon1, Canon2)) {
        if (Keep == Canon1)
          continue; // Keep the first path's.
        Result =
            ContextFactory.add(ContextFactory.remove(Result, Dec), Dec, *I2);
        continue;
      }
      // A phi merged with a definition it already covers is just the phi:
      // at a join of three or more predecessors the paths merge pairwise,
      // so e.g. (constant, call) -> phi followed by (phi, constant) must
      // not discard the merge the first pair created; absorbing a
      // value-equal constant that is not an operand keeps the result
      // independent of the order in which the paths merge (phiAbsorbs()).
      if (phiAbsorbs(Dec, Canon1, Canon2))
        continue; // Keep the first path's phi.
      if (phiAbsorbs(Dec, Canon2, Canon1)) {
        // Keep the second path's phi.
        Result =
            ContextFactory.add(ContextFactory.remove(Result, Dec), Dec, *I2);
        continue;
      }
      // The underlying definitions differ. If both are known (and not
      // already merges themselves), remember the pair as a phi definition;
      // otherwise invalidate.
      if (Canon1 != 0 && Canon2 != 0 && !VarDefinitions[Canon1].isPhi() &&
          !VarDefinitions[Canon2].isPhi())
        Result = addPhiDefinition(Dec, P.second, *I2, Result);
      else
        Result = clearDefinition(Dec, Result);
    }
  }
  return Result;
}

// For every variable in C, create a new variable that refers to the
// definition in C.  Return a new context that contains these new variables.
// (We use this for a naive implementation of SSA on loop back-edges.)
LocalVariableMap::Context LocalVariableMap::createReferenceContext(Context C) {
  Context Result = getEmptyContext();
  for (const auto &P : C)
    Result = addReference(P.first, P.second, Result);
  return Result;
}

// This routine also takes the intersection of C1 and C2, but it does so by
// altering the VarDefinitions.  C1 must be the result of an earlier call to
// createReferenceContext.
void LocalVariableMap::intersectBackEdge(Context C1, Context C2) {
  for (const auto &P : C1) {
    const unsigned I1 = P.second;
    VarDefinition *VDef = &VarDefinitions[I1];
    assert(VDef->isReference() || VDef->isPhi());

    const unsigned *I2 = C2.lookup(P.first);
    if (!I2) {
      // Variable does not exist at the end of the loop, invalidate.
      VDef->invalidateRef();
      continue;
    }

    const unsigned Canon2 = getCanonicalDefinitionID(*I2);

    if (VDef->isPhi()) {
      // A previous back edge already merged this variable. Keep the phi only
      // if this back edge carries a definition the merge already covers --
      // one of its operands, or a value-equal constant whose chain avoids
      // the non-constant operand up to the loop head (phiAbsorbs(), as at
      // branch joins) -- or the loop-head reference itself (Canon2 == I1, a
      // phi is its own canonical): a back edge that does not reassign the
      // variable, or reassigns it a value the merge covers, must not
      // discard the merge another back edge created.
      if (Canon2 != I1 && !phiAbsorbs(P.first, I1, Canon2, /*StopAt=*/I1))
        VDef->invalidateRef();
      continue;
    }

    // Compare the canonical IDs. This correctly handles chains of references
    // and determines if the variable is truly loop-invariant.
    if (VDef->CanonicalRef != Canon2) {
      // The variable was reassigned in the loop a value that is a constant
      // value-equal to the loop-head value: interchangeable for resolution
      // purposes (valueEqualConstants(), as at branch joins), so the head
      // reference stands.
      if (valueEqualConstants(VDef->CanonicalRef, Canon2))
        continue;
      // The variable is redefined in the loop. The back-edge value may
      // itself be a merge created at an intra-loop join with the loop-head
      // value as one of its operands (e.g. the path of a `continue` that
      // reassigned the variable joining the loop-end path that did not):
      // the head value then merges with the other operand. An operand that
      // is a constant value-equal to the head's constant stands in for the
      // head value the same way -- provided its chain avoids the other
      // operand up to the loop head, exactly as absorbing it at a branch
      // join would require (phiAbsorbs()).
      auto RefersTo = [this](unsigned ID, unsigned Target) {
        while (ID > 0 && ID != Target && VarDefinitions[ID].isReference())
          ID = VarDefinitions[ID].DirectRef;
        return ID == Target;
      };
      unsigned Alt = *I2;
      unsigned CanonAlt = Canon2;
      if (!TracksTryAcquires)
        Alt = CanonAlt = 0; // No merge is ever read: invalidate (below).
      if (CanonAlt != 0 && VarDefinitions[CanonAlt].isPhi()) {
        const VarDefinition &P2 = VarDefinitions[CanonAlt];
        const unsigned OpD = getCanonicalDefinitionID(P2.DirectRef);
        const unsigned OpA = getCanonicalDefinitionID(P2.PhiAlt);
        if (RefersTo(P2.DirectRef, I1))
          Alt = P2.PhiAlt;
        else if (RefersTo(P2.PhiAlt, I1))
          Alt = P2.DirectRef;
        else if (valueEqualConstants(VDef->CanonicalRef, OpD) &&
                 chainAvoids(P.first, OpD, OpA, /*StopAt=*/I1))
          Alt = P2.PhiAlt;
        else if (valueEqualConstants(VDef->CanonicalRef, OpA) &&
                 chainAvoids(P.first, OpA, OpD, /*StopAt=*/I1))
          Alt = P2.DirectRef;
        else
          Alt = 0;
        CanonAlt = getCanonicalDefinitionID(Alt);
      }
      // If both the incoming definition and the back edge's are known (and
      // the latter is not itself a merge), remember the pair as a phi
      // definition (in place, like the invalidation below) rather than
      // discarding it, so that a branch on a try-acquire result merged
      // with its pre-loop initializer can still be resolved.
      if (VDef->CanonicalRef != 0 && Alt != 0 && CanonAlt != 0 &&
          !VarDefinitions[CanonAlt].isPhi()) {
        VDef->PhiAlt = Alt;
        VDef->CanonicalRef = 0;
      } else {
        VDef->invalidateRef(); // Mark this variable as undefined
      }
      // A back edge is the only thing that ever changes an existing
      // definition (in place, above), which can make a memoized clean
      // chain of this variable, and any memoized chain walk, stale. The
      // walks are keyed by definition and a changed definition can be
      // anywhere in another one's chain, so they all go.
      CleanChains.erase(P.first);
      ChainDefs.clear();
    }
  }
}

// Traverse the CFG in topological order, so all predecessors of a block
// (excluding back-edges) are visited before the block itself.  At
// each point in the code, we calculate a Context, which holds the set of
// variable definitions which are visible at that point in execution.
// Visible variables are mapped to their definitions using an array that
// contains all definitions.
//
// At join points in the CFG, the set is computed as the intersection of
// the incoming sets along each edge, E.g.
//
//                       { Context                 | VarDefinitions }
//   int x = 0;          { x -> x1                 | x1 = 0 }
//   int y = 0;          { x -> x1, y -> y1        | y1 = 0, x1 = 0 }
//   if (b) x = 1;       { x -> x2, y -> y1        | x2 = 1, y1 = 0, ... }
//   else   x = 2;       { x -> x3, y -> y1        | x3 = 2, x2 = 1, ... }
//   ...                 { y -> y1  (x is unknown) | x3 = 2, x2 = 1, ... }
//
// This is essentially a simpler and more naive version of the standard SSA
// algorithm.  Those definitions that remain in the intersection are from blocks
// that strictly dominate the current block.  We do not bother to insert proper
// phi nodes, because they are not used in our analysis; instead, wherever
// a phi node would be required, we simply remove that definition from the
// context (E.g. x above).
//
// The initial traversal does not capture back-edges, so those need to be
// handled on a separate pass.  Whenever the first pass encounters an
// incoming back edge, it duplicates the context, creating new definitions
// that refer back to the originals.  (These correspond to places where SSA
// might have to insert a phi node.)  On the second pass, these definitions are
// set to NULL if the variable has changed on the back-edge (i.e. a phi
// node was actually required.)  E.g.
//
//                       { Context           | VarDefinitions }
//   int x = 0, y = 0;   { x -> x1, y -> y1  | y1 = 0, x1 = 0 }
//   while (b)           { x -> x2, y -> y1  | [1st:] x2=x1; [2nd:] x2=NULL; }
//     x = x+1;          { x -> x3, y -> y1  | x3 = x2 + 1, ... }
//   ...                 { y -> y1           | x3 = 2, x2 = 1, ... }
void LocalVariableMap::traverseCFG(AnalysisDeclContext &AC, CFG *CFGraph,
                                   const PostOrderCFGView *SortedGraph,
                                   std::vector<CFGBlockInfo> &BlockInfo,
                                   bool BetaWarnings) {
  PostOrderCFGView::CFGBlockSet VisitedBlocks(CFGraph);

  // Whether any try-acquire is called (or constructed) at all (see
  // TracksTryAcquires).
  TracksTryAcquires = llvm::any_of(*CFGraph, [](const CFGBlock *B) {
    return llvm::any_of(*B, [](const CFGElement &E) {
      std::optional<CFGStmt> CS = E.getAs<CFGStmt>();
      if (!CS)
        return false;
      const Decl *Callee = nullptr;
      if (const auto *CE = dyn_cast<CallExpr>(CS->getStmt()))
        Callee = CE->getCalleeDecl();
      else if (const auto *CtorE = dyn_cast<CXXConstructExpr>(CS->getStmt()))
        Callee = CtorE->getConstructor();
      return Callee && Callee->hasAttr<TryAcquireCapabilityAttr>();
    });
  });
  ReadsDefinitions = TracksTryAcquires || BetaWarnings;

  for (const auto *CurrBlock : *SortedGraph) {
    unsigned CurrBlockID = CurrBlock->getBlockID();
    CFGBlockInfo *CurrBlockInfo = &BlockInfo[CurrBlockID];

    VisitedBlocks.insert(CurrBlock);

    // Calculate the entry context for the current block
    bool HasBackEdges = false;
    bool CtxInit = true;
    for (CFGBlock::const_pred_iterator PI = CurrBlock->pred_begin(),
         PE  = CurrBlock->pred_end(); PI != PE; ++PI) {
      // if *PI -> CurrBlock is a back edge, so skip it
      if (*PI == nullptr || !VisitedBlocks.alreadySet(*PI)) {
        HasBackEdges = true;
        continue;
      }

      unsigned PrevBlockID = (*PI)->getBlockID();
      CFGBlockInfo *PrevBlockInfo = &BlockInfo[PrevBlockID];

      if (CtxInit) {
        CurrBlockInfo->EntryContext = PrevBlockInfo->ExitContext;
        CtxInit = false;
      }
      else {
        CurrBlockInfo->EntryContext =
          intersectContexts(CurrBlockInfo->EntryContext,
                            PrevBlockInfo->ExitContext);
      }
    }

    // Duplicate the context if we have back-edges, so we can call
    // intersectBackEdges later.
    if (HasBackEdges)
      CurrBlockInfo->EntryContext =
        createReferenceContext(CurrBlockInfo->EntryContext);

    // Create a starting context index for the current block
    saveContext(nullptr, CurrBlockInfo->EntryContext);
    CurrBlockInfo->EntryIndex = getContextIndex();

    // Visit all the statements in the basic block.
    VarMapBuilder VMapBuilder(this, CurrBlockInfo->EntryContext, AC);
    VMapBuilder.CurBlock = CurrBlock;
    for (const auto &BI : *CurrBlock) {
      switch (BI.getKind()) {
        case CFGElement::Statement: {
          CFGStmt CS = BI.castAs<CFGStmt>();
          VMapBuilder.Visit(CS.getStmt());
          break;
        }
        default:
          break;
      }
    }
    CurrBlockInfo->ExitContext = VMapBuilder.Ctx;

    // Mark variables on back edges as "unknown" if they've been changed.
    for (CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin(),
         SE  = CurrBlock->succ_end(); SI != SE; ++SI) {
      // if CurrBlock -> *SI is *not* a back edge
      if (*SI == nullptr || !VisitedBlocks.alreadySet(*SI))
        continue;

      CFGBlock *FirstLoopBlock = *SI;
      Context LoopBegin = BlockInfo[FirstLoopBlock->getBlockID()].EntryContext;
      Context LoopEnd   = CurrBlockInfo->ExitContext;
      intersectBackEdge(LoopBegin, LoopEnd);
    }
  }

  // Put an extra entry at the end of the indexed context array
  unsigned exitID = CFGraph->getExit().getBlockID();
  saveContext(nullptr, BlockInfo[exitID].ExitContext);
}

/// Find the appropriate source locations to use when producing diagnostics for
/// each block in the CFG.
static void findBlockLocations(CFG *CFGraph,
                               const PostOrderCFGView *SortedGraph,
                               std::vector<CFGBlockInfo> &BlockInfo) {
  for (const auto *CurrBlock : *SortedGraph) {
    CFGBlockInfo *CurrBlockInfo = &BlockInfo[CurrBlock->getBlockID()];

    // Find the source location of the last statement in the block, if the
    // block is not empty.
    if (const Stmt *S = CurrBlock->getTerminatorStmt()) {
      CurrBlockInfo->EntryLoc = CurrBlockInfo->ExitLoc = S->getBeginLoc();
    } else {
      for (CFGBlock::const_reverse_iterator BI = CurrBlock->rbegin(),
           BE = CurrBlock->rend(); BI != BE; ++BI) {
        // FIXME: Handle other CFGElement kinds.
        if (std::optional<CFGStmt> CS = BI->getAs<CFGStmt>()) {
          CurrBlockInfo->ExitLoc = CS->getStmt()->getBeginLoc();
          break;
        }
      }
    }

    if (CurrBlockInfo->ExitLoc.isValid()) {
      // This block contains at least one statement. Find the source location
      // of the first statement in the block.
      for (const auto &BI : *CurrBlock) {
        // FIXME: Handle other CFGElement kinds.
        if (std::optional<CFGStmt> CS = BI.getAs<CFGStmt>()) {
          CurrBlockInfo->EntryLoc = CS->getStmt()->getBeginLoc();
          break;
        }
      }
    } else if (CurrBlock->pred_size() == 1 && *CurrBlock->pred_begin() &&
               CurrBlock != &CFGraph->getExit()) {
      // The block is empty, and has a single predecessor. Use its exit
      // location.
      CurrBlockInfo->EntryLoc = CurrBlockInfo->ExitLoc =
          BlockInfo[(*CurrBlock->pred_begin())->getBlockID()].ExitLoc;
    } else if (CurrBlock->succ_size() == 1 && *CurrBlock->succ_begin()) {
      // The block is empty, and has a single successor. Use its entry
      // location.
      CurrBlockInfo->EntryLoc = CurrBlockInfo->ExitLoc =
          BlockInfo[(*CurrBlock->succ_begin())->getBlockID()].EntryLoc;
    }
  }
}

namespace {

static void installNegativeFact(FactSet &FSet, FactManager &FactMan,
                                const CapabilityExpr &NegCp, SourceLocation Loc,
                                bool KeepExisting);

class LockableFactEntry final : public FactEntry {
private:
  /// Reentrancy depth: incremented when a capability has been acquired
  /// again after its initial acquisition -- by a reentrant acquire, or by
  /// the resolved success of a try-acquire over a definite hold.
  unsigned int ReentrancyDepth = 0;

  LockableFactEntry(const CapabilityExpr &CE, LockKind LK, SourceLocation Loc,
                    SourceKind Src)
      : FactEntry(Lockable, CE, LK, Loc, Src) {}

public:
  static LockableFactEntry *create(llvm::BumpPtrAllocator &Alloc,
                                   const LockableFactEntry &Other) {
    return new (Alloc) LockableFactEntry(Other);
  }

  static LockableFactEntry *create(llvm::BumpPtrAllocator &Alloc,
                                   const CapabilityExpr &CE, LockKind LK,
                                   SourceLocation Loc,
                                   SourceKind Src = Acquired) {
    return new (Alloc) LockableFactEntry(CE, LK, Loc, Src);
  }

  unsigned int getReentrancyDepth() const { return ReentrancyDepth; }

  void
  handleRemovalFromIntersection(const FactSet &FSet, FactManager &FactMan,
                                SourceLocation JoinLoc, LockErrorKind LEK,
                                ThreadSafetyHandler &Handler) const override {
    if (lossNeedsWarning()) {
      Handler.handleMutexHeldEndOfScope(getKind(), toString(), loc(), JoinLoc,
                                        LEK);
    }
  }

  void handleLock(FactSet &FSet, FactManager &FactMan, const FactEntry &entry,
                  ThreadSafetyHandler &Handler) const override {
    if (const FactEntry *RFact = attemptReenter(FactMan, entry.kind())) {
      // This capability has been reentrantly acquired.
      FSet.replaceFact(FactMan, *this, RFact);
    } else {
      Handler.handleDoubleLock(entry.getKind(), entry.toString(), loc(),
                               entry.loc(), false);
    }
  }

  void handleUnlock(FactSet &FSet, FactManager &FactMan,
                    const CapabilityExpr &Cp, SourceLocation UnlockLoc,
                    bool FullyRemove,
                    ThreadSafetyHandler &Handler) const override {
    FSet.removeFact(FactMan, *this);

    if (const FactEntry *RFact = leaveReentrant(FactMan)) {
      // This capability remains reentrantly acquired.
      FSet.addLock(FactMan, RFact);
      return;
    }
    // The hold is gone: whatever try-acquire's success proved a level of
    // it no longer proves a live one -- its stored result is stale (see
    // FactSet::releaseProved()).
    FSet.releaseProved(FactMan, Cp, UnlockLoc);
    if (!Cp.negative() && !FSet.anyConditional(FactMan, Cp)) {
      // Provably released -- unless a conditional try fact remains, in
      // which case the capability is now back to conditionally held.
      installNegativeFact(FSet, FactMan, !Cp, UnlockLoc,
                          /*KeepExisting=*/false);
    }
  }

  // Return an updated FactEntry one level deeper, or nullptr if a blocking
  // acquisition cannot nest in this capability: it must be reentrant, and
  // the kinds must match.
  const FactEntry *attemptReenter(FactManager &FactMan,
                                  LockKind ReenterKind) const {
    if (!reentrant())
      return nullptr;
    if (kind() != ReenterKind)
      return nullptr;
    auto *NewFact = FactMan.createFact<LockableFactEntry>(*this);
    NewFact->ReentrancyDepth++;
    return NewFact;
  }

  // Return an updated FactEntry if we are releasing a capability previously
  // acquired more than once, nullptr otherwise.
  const FactEntry *leaveReentrant(FactManager &FactMan) const {
    if (!ReentrancyDepth)
      return nullptr;
    auto *NewFact = FactMan.createFact<LockableFactEntry>(*this);
    NewFact->ReentrancyDepth--;
    return NewFact;
  }

  /// This definite fact one level deeper: the level the success of a
  /// try-acquire proved (a try fact resolved over this hold). Whatever the
  /// capability's reentrancy: at runtime a try-acquire over a held
  /// capability fails rather than deadlocks, so its success edge is merely
  /// dead code that must still be well-formed.
  const LockableFactEntry *deepen(FactManager &FactMan) const {
    auto *NewFact = FactMan.createFact<LockableFactEntry>(*this);
    NewFact->ReentrancyDepth++;
    return NewFact;
  }

  /// A conditional try fact of this capability -- kind, source and location
  /// as this fact's -- speaking about \p Origin: the form a definite hold
  /// takes when a join can only keep it as "held if \p Origin succeeded".
  const TryFactEntry *asConditional(FactManager &FactMan,
                                    const Expr *Origin) const {
    return FactMan.createFact<TryFactEntry>(*this, kind(), loc(), source(),
                                            Origin);
  }

  static bool classof(const FactEntry *A) {
    return A->getFactEntryKind() == Lockable;
  }
};

const LockableFactEntry *TryFactEntry::asDefinite(FactManager &FactMan) const {
  return FactMan.createFact<LockableFactEntry>(*this, kind(), loc(), source());
}

/// Install the negative fact for \p NegCp at \p Loc: the capability is
/// provably not held from here. An existing negative is kept when
/// \p KeepExisting (it already proves as much) and replaced otherwise, so
/// that the fact records the latest release.
static void installNegativeFact(FactSet &FSet, FactManager &FactMan,
                                const CapabilityExpr &NegCp, SourceLocation Loc,
                                bool KeepExisting) {
  FactSet::iterator Existing = FSet.findDefiniteIter(FactMan, NegCp);
  if (Existing != FSet.end() && KeepExisting)
    return;
  auto *NegFact =
      FactMan.createFact<LockableFactEntry>(NegCp, LK_Exclusive, Loc);
  // Replacing in place keeps the superseded fact's slot: removing swaps in
  // the set's last element and would reorder unrelated facts.
  if (Existing != FSet.end())
    FSet.replaceFact(FactMan, Existing, NegFact);
  else
    FSet.addLock(FactMan, NegFact);
}

/// The location for an unmatched-unlock "released here" note: the negative
/// fact's location if one exists -- unless it came from a try-acquire's
/// failure edge (getEdgeLockset()), which records where the call failed,
/// not a release, and the note would misread it; nor from a try-release
/// call's success edge, where it names the call rather than a release. A
/// negative fact a branch recorded sits at its call's location
/// (FactManager::isTryAcquireLoc()).
static SourceLocation unmatchedUnlockNoteLoc(const FactSet &FSet,
                                             FactManager &FactMan,
                                             const CapabilityExpr &Cp) {
  if (const FactEntry *Neg = FSet.findDefinite(FactMan, !Cp);
      Neg && !FactMan.isTryAcquireLoc(Neg->loc(), Cp))
    return Neg->loc();
  // No negative fact survived the joins: the release that invalidated a
  // try-acquire's result, recorded on its try fact, names it instead (the
  // latest such try fact, in insertion order).
  SourceLocation Loc;
  for (const auto &Fact : FSet) {
    const auto *W = dyn_cast<TryFactEntry>(&FactMan[Fact]);
    if (W && (W->released() || W->mayBeReleased()) &&
        W->releaseLoc().isValid() && W->matches(Cp))
      Loc = W->releaseLoc();
  }
  return Loc;
}

/// Release the capability \p Cp, which is only conditionally held (conditional
/// try facts but no definite fact); returns true if the release was handled
/// here. With a \p Handler, diagnose like an unmatched unlock, drop every
/// conditional try fact, and leave the negative fact behind: the release is
/// an unconditional demand, and the thread provably does not hold the
/// capability afterwards, whether the try-acquires succeeded or failed. A
/// null \p Handler (a scoped guard's destructor, FullyRemove=true) is a
/// conditional release -- the destructor releases the capability only if
/// the guard holds it -- so the guard's own conditional try fact, the one
/// its construction \p OwnOrigin created, is disarmed silently: the
/// conditional release pairs with it exactly, discharging the obligation to
/// check the result and, with no other hold of the capability left, leaving
/// the negative fact. Another call's try fact is kept unchanged: it records
/// an acquisition the guard does not own, which the destructor's
/// conditional release cannot pair with.
static bool handleUncheckedConditionalUnlock(FactSet &FSet,
                                             FactManager &FactMan,
                                             const CapabilityExpr &Cp,
                                             SourceLocation UnlockLoc,
                                             ThreadSafetyHandler *Handler) {
  if (!FSet.anyConditional(FactMan, Cp))
    return false;
  if (!Handler)
    return true;
  Handler->handleUnmatchedUnlock(Cp.getKind(), Cp.toString(), UnlockLoc,
                                 SourceLocation(), true);
  FSet.releaseConditional(FactMan, Cp, UnlockLoc);
  // A pre-existing negative already proves not-held on every path and is
  // kept.
  if (!Cp.negative())
    installNegativeFact(FSet, FactMan, !Cp, UnlockLoc,
                        /*KeepExisting=*/true);
  return true;
}

enum UnderlyingCapabilityKind {
  UCK_Acquired,          ///< Any kind of acquired capability.
  UCK_ReleasedShared,    ///< Shared capability that was released.
  UCK_ReleasedExclusive, ///< Exclusive capability that was released.
};

struct UnderlyingCapability {
  CapabilityExpr Cap;
  UnderlyingCapabilityKind Kind;
};

/// Install the definite fact \p Entry for a capability held only
/// conditionally, through the try facts of unresolved try-acquires (\p Cond
/// is the first of them). The definite level goes beside the conditional
/// ones of its own kind: a reentrant acquire nests in them silently, while
/// any other may deadlock exactly when a try-acquire succeeded, which
/// \p Handler diagnoses -- and the code after it is still verified as if
/// the try-acquire were live, its try fact resolving a deeper level on the
/// branch on its result (as a try-acquire over a definite hold does), or
/// diagnosed where it is lost. A conditional try fact gives way to the
/// definite level instead in two cases: when any of them is of the other lock
/// kind, since none can then be a level of this hold, all of them go; and the
/// acquiring scoped object's own (\p OwnOrigin, its construction), which no
/// branch can ever resolve and whose destructor releases the definite level in
/// its place. The one policy for a blocking acquisition over a conditional
/// hold, whether the acquisition is a call's (ThreadSafetyAnalyzer::addLock())
/// or a scoped object's (ScopedLockableFactEntry::lock()).
static void addLockOverConditional(FactSet &FSet, FactManager &FactMan,
                                   const FactEntry *Entry,
                                   const FactEntry &Cond,
                                   ThreadSafetyHandler *Handler,
                                   const Expr *OwnOrigin = nullptr) {
  const bool SameKind =
      FSet.conditionalsAllOfKind(FactMan, *Entry, Entry->kind());
  // A reentrant capability nests silently: the acquisition adds a level to a
  // hold that may already have one, which is exactly what reentrancy means.
  const bool SilentNest =
      Entry->reentrant() && isa<LockableFactEntry>(Entry) && SameKind;
  if (Handler && !SilentNest)
    Handler->handleDoubleLock(Entry->getKind(), Entry->toString(), Cond.loc(),
                              Entry->loc(), /*MaybeHeld=*/true);
  if (!SameKind)
    FSet.removeAllConditional(FactMan, *Entry);
  else if (OwnOrigin)
    // The acquiring scope's own conditional level gives way to the definite
    // one it takes in its place, which its destructor releases instead: the
    // guard owns one level at a time, so a release of it is unambiguous.
    FSet.removeConditionalsOf(FactMan, *Entry, OwnOrigin);
  // The acquisition consumes the negative fact, as a fresh one does: the
  // capability is held now, whatever the try-acquire beside it did.
  FSet.removeDefinite(FactMan, !*Entry);
  FSet.addLock(FactMan, Entry);
}

class ScopedLockableFactEntry final
    : public FactEntry,
      private llvm::TrailingObjects<ScopedLockableFactEntry,
                                    UnderlyingCapability> {
  friend TrailingObjects;

private:
  const unsigned ManagedCapacity;
  unsigned ManagedSize = 0;
  /// The construction whose try-acquire attributes acquired the managed
  /// capabilities conditionally, if any: the origin of the try facts this
  /// guard created, which are exactly the ones its destructor releases
  /// (see unlock()). Null for a definite guard, which is what tells the two
  /// apart: a definite guard's level is a hold it created, while a try-guard
  /// may merely nest in a hold that is not its own to release.
  const Expr *CondAcquireExpr = nullptr;

  ScopedLockableFactEntry(const CapabilityExpr &CE, SourceLocation Loc,
                          SourceKind Src, unsigned ManagedCapacity)
      : FactEntry(ScopedLockable, CE, LK_Exclusive, Loc, Src),
        ManagedCapacity(ManagedCapacity) {}

  void addManaged(const CapabilityExpr &M, UnderlyingCapabilityKind UCK) {
    assert(ManagedSize < ManagedCapacity);
    new (getTrailingObjects() + ManagedSize) UnderlyingCapability{M, UCK};
    ++ManagedSize;
  }

public:
  /// The capabilities this scope acquired or released, which its members and
  /// its destructor act on in its place.
  ArrayRef<UnderlyingCapability> getManaged() const {
    return getTrailingObjects(ManagedSize);
  }

  static ScopedLockableFactEntry *create(llvm::BumpPtrAllocator &Alloc,
                                         const CapabilityExpr &CE,
                                         SourceLocation Loc, SourceKind Src,
                                         unsigned ManagedCapacity) {
    void *Storage =
        Alloc.Allocate(totalSizeToAlloc<UnderlyingCapability>(ManagedCapacity),
                       alignof(ScopedLockableFactEntry));
    return new (Storage) ScopedLockableFactEntry(CE, Loc, Src, ManagedCapacity);
  }

  CapExprSet getUnderlyingMutexes() const {
    CapExprSet UnderlyingMutexesSet;
    for (const UnderlyingCapability &UnderlyingMutex : getManaged())
      UnderlyingMutexesSet.push_back(UnderlyingMutex.Cap);
    return UnderlyingMutexesSet;
  }

  /// \name Adding managed locks
  /// Capacity for managed locks must have been allocated via \ref create.
  /// There is no reallocation in case the capacity is exceeded!
  /// \{
  void addLock(const CapabilityExpr &M) { addManaged(M, UCK_Acquired); }

  /// Record that this guard's construction \p Exp acquired its managed
  /// capabilities conditionally.
  void setCondAcquireExpr(const Expr *Exp) { CondAcquireExpr = Exp; }

  void addExclusiveUnlock(const CapabilityExpr &M) {
    addManaged(M, UCK_ReleasedExclusive);
  }

  void addSharedUnlock(const CapabilityExpr &M) {
    addManaged(M, UCK_ReleasedShared);
  }
  /// \}

  void
  handleRemovalFromIntersection(const FactSet &FSet, FactManager &FactMan,
                                SourceLocation JoinLoc, LockErrorKind LEK,
                                ThreadSafetyHandler &Handler) const override {
    if (isEndOfFunctionLEK(LEK))
      return;

    for (const auto &UnderlyingMutex : getManaged()) {
      // Held or possibly held: either way the scope still has it to release.
      const auto *Entry =
          FSet.findDefiniteOrConditional(FactMan, UnderlyingMutex.Cap);
      if ((UnderlyingMutex.Kind == UCK_Acquired && Entry) ||
          (UnderlyingMutex.Kind != UCK_Acquired && !Entry)) {
        // If this scoped lock manages another mutex, and if the underlying
        // mutex is still/not held, then warn about the underlying mutex.
        Handler.handleMutexHeldEndOfScope(UnderlyingMutex.Cap.getKind(),
                                          UnderlyingMutex.Cap.toString(), loc(),
                                          JoinLoc, LEK);
      }
    }
  }

  void handleLock(FactSet &FSet, FactManager &FactMan, const FactEntry &entry,
                  ThreadSafetyHandler &Handler) const override {
    for (const auto &UnderlyingMutex : getManaged()) {
      if (UnderlyingMutex.Kind == UCK_Acquired)
        lock(FSet, FactMan, UnderlyingMutex.Cap, entry.kind(), entry.loc(),
             &Handler);
      else
        unlock(FSet, FactMan, UnderlyingMutex.Cap, entry.loc(), &Handler);
    }
  }

  void handleUnlock(FactSet &FSet, FactManager &FactMan,
                    const CapabilityExpr &Cp, SourceLocation UnlockLoc,
                    bool FullyRemove,
                    ThreadSafetyHandler &Handler) const override {
    assert(!Cp.negative() && "Managing object cannot be negative.");
    for (const auto &UnderlyingMutex : getManaged()) {
      // Remove/lock the underlying mutex if it exists/is still unlocked; warn
      // on double unlocking/locking if we're not destroying the scoped object.
      ThreadSafetyHandler *TSHandler = FullyRemove ? nullptr : &Handler;
      if (UnderlyingMutex.Kind == UCK_Acquired) {
        unlock(FSet, FactMan, UnderlyingMutex.Cap, UnlockLoc, TSHandler);
      } else {
        LockKind kind = UnderlyingMutex.Kind == UCK_ReleasedShared
                            ? LK_Shared
                            : LK_Exclusive;
        lock(FSet, FactMan, UnderlyingMutex.Cap, kind, UnlockLoc, TSHandler);
      }
    }
    if (FullyRemove)
      FSet.removeFact(FactMan, *this);
  }

  static bool classof(const FactEntry *A) {
    return A->getFactEntryKind() == ScopedLockable;
  }

private:
  void lock(FactSet &FSet, FactManager &FactMan, const CapabilityExpr &Cp,
            LockKind kind, SourceLocation loc,
            ThreadSafetyHandler *Handler) const {
    if (const auto It = FSet.findDefiniteIter(FactMan, Cp); It != FSet.end()) {
      const auto &Fact = cast<LockableFactEntry>(FactMan[*It]);
      if (const FactEntry *RFact = Fact.attemptReenter(FactMan, kind)) {
        // This capability has been reentrantly acquired.
        FSet.replaceFact(FactMan, It, RFact);
      } else if (Handler) {
        Handler->handleDoubleLock(Cp.getKind(), Cp.toString(), Fact.loc(), loc,
                                  /*MaybeHeld=*/false);
      }
      return;
    }
    if (const FactEntry *Cond = FSet.firstConditional(FactMan, Cp)) {
      addLockOverConditional(
          FSet, FactMan,
          FactMan.createFact<LockableFactEntry>(Cp, kind, loc, Managed), *Cond,
          Handler, CondAcquireExpr);
      return;
    }
    // The acquisition consumes the negative fact: the capability is held now.
    FSet.removeDefinite(FactMan, !Cp);
    FSet.addLock(FactMan,
                 FactMan.createFact<LockableFactEntry>(Cp, kind, loc, Managed));
  }

  void unlock(FactSet &FSet, FactManager &FactMan, const CapabilityExpr &Cp,
              SourceLocation loc, ThreadSafetyHandler *Handler) const {
    const FactEntry *Def = FSet.findDefinite(FactMan, Cp);
    // The level this guard releases is the one it acquired. While its own
    // acquisition is still conditional -- and it has not since acquired a
    // definite level of its own, which would be the level to release first --
    // the try fact is what is spent: the guard's death disarms it silently,
    // since the destructor releases the capability only if the guard holds
    // it, which pairs exactly with the conditional acquisition, while an
    // explicit release member is an unconditional demand and warns that the
    // capability may not be held. Either way a hold beside it survives: an
    // outer definite hold the guard did not acquire, and a conditional level
    // another call contributed, which is left to that call.
    if (CondAcquireExpr &&
        FSet.removeConditionalsOf(FactMan, Cp, CondAcquireExpr)) {
      if (Handler)
        Handler->handleUnmatchedUnlock(Cp.getKind(), Cp.toString(), loc,
                                       SourceLocation(), /*MaybeHeld=*/true);
      if (!Def && !FSet.anyConditional(FactMan, Cp) && !Cp.negative())
        installNegativeFact(FSet, FactMan, !Cp, loc, /*KeepExisting=*/true);
      return;
    }
    if (const auto It = FSet.findDefiniteIter(FactMan, Cp); It != FSet.end()) {
      const auto &Fact = cast<LockableFactEntry>(FactMan[*It]);
      // A try-guard whose own conditional level is gone -- spent above, or
      // subsumed by a blocking acquire -- has no claim on a hold it did not
      // create: releasing one it merely nests in would release another
      // acquisition's level.
      if (CondAcquireExpr && !Fact.managed() && !Fact.asserted()) {
        if (Handler)
          Handler->handleUnmatchedUnlock(Cp.getKind(), Cp.toString(), loc,
                                         SourceLocation(), /*MaybeHeld=*/true);
        return;
      }
      if (const FactEntry *RFact = Fact.leaveReentrant(FactMan)) {
        // This capability remains reentrantly acquired.
        FSet.replaceFact(FactMan, It, RFact);
        return;
      }

      // As in LockableFactEntry::handleUnlock(): released -- unless a
      // conditional try fact remains, in which case the capability is now
      // merely conditionally held -- and releasing a hold proved by a
      // try-acquire's success leaves the call's stored result stale.
      FSet.erase(It);
      FSet.releaseProved(FactMan, Cp, loc);
      if (!FSet.anyConditional(FactMan, Cp))
        installNegativeFact(FSet, FactMan, !Cp, loc,
                            /*KeepExisting=*/false);
      return;
    }
    if (handleUncheckedConditionalUnlock(FSet, FactMan, Cp, loc, Handler))
      return;
    if (Handler)
      Handler->handleUnmatchedUnlock(Cp.getKind(), Cp.toString(), loc,
                                     unmatchedUnlockNoteLoc(FSet, FactMan, Cp),
                                     false);
  }
};

/// Per-switch facts getSwitchEdgeValue() needs on every outgoing edge,
/// computed once per terminator (getSwitchSummary()): the switch's own
/// case labels with their value ranges -- what the default edge excludes
/// the condition from -- whether those cover zero and one, and whether
/// the condition is boolean.
struct SwitchSummary {
  /// This switch's own case labels with their evaluated [Lo, Hi] value
  /// range: the fall-out successor can carry a label belonging to an
  /// enclosing switch, which says nothing about this condition.
  llvm::SmallDenseMap<const CaseStmt *, std::pair<llvm::APSInt, llvm::APSInt>,
                      8>
      OwnCases;
  bool ZeroListed = false;
  bool OneListed = false;
  bool IsBool = false;
};

/// State for a join of two locksets.
struct JoinContext {
  /// The location of the join point, for error reporting.
  SourceLocation JoinLoc;
  /// The warning if a mutex is missing from the entry set.
  LockErrorKind EntryLEK;
  /// The warning if a mutex is missing from the exit set.
  LockErrorKind ExitLEK;

  /// The try-acquire call whose result the joining block's terminator branches
  /// on, if any. A held/conditionally held difference between facts that both
  /// speak about that call is not diagnosed as a lost hold: the paths
  /// re-diverge at the terminator, so the merged state keeps the call's
  /// conditional try fact (any reentrancy depth is diagnosed but kept) and
  /// re-resolves it on the outgoing edges.
  const Expr *RebranchTryLock = nullptr;
  /// Whether every outgoing path of the joining block reaches the branch
  /// on \c RebranchTryLock's result (false when the branch was found
  /// behind a short-circuit, whose other edge escapes unresolved). When
  /// false, weakening a definitely-held fact is diagnosed at the join
  /// after all -- the exemption's promise of re-resolution does not hold
  /// on the escaping paths -- though the fact is still demoted so the
  /// paths that do rebranch resolve it.
  bool RebranchResolvesAllPaths = true;

  /// Record what the branch below \p Call resolves. The one setter both
  /// join sites use: a branch join and a loop back edge ask the same
  /// question of the block whose condition re-resolves the facts, so they
  /// must record the same answer -- a short-circuit escape that leaves the
  /// result unresolved defeats the exemption at either.
  template <typename CallT> void setRebranch(const CallT &Call) {
    RebranchTryLock = Call.TrylockCall;
    RebranchMergedCall = Call.MergedCall;
    RebranchResolvesAllPaths = Call.ResolvesAllPaths;
  }
  /// When the branched-on variable merges the results of two structurally
  /// identical try-acquire calls (the merge resolves to \c RebranchTryLock,
  /// the first path's call), the second path's call. A side holding one
  /// of the two calls' try facts alone holds "the result of that call"
  /// either way, so its try fact folds into the try fact of
  /// \c RebranchTryLock, which the outgoing edges resolve like a single
  /// call's.
  const Expr *RebranchMergedCall = nullptr;
  /// For a loop join under -Wthread-safety-beta: the try-acquire calls
  /// whose results are branched on somewhere inside the loop, and so are
  /// (or will be, on the next iteration) checked around it. A conditional
  /// try fact of any other call reaching the back edge is re-executed or
  /// discarded unchecked (LocksetJoin::joinTryFactFromExit()). Null where
  /// the information is not computed, which exempts every try fact.
  const llvm::SmallPtrSetImpl<const Expr *> *CheckedAroundLoop = nullptr;
  /// The entry set belongs to an already-analyzed loop head (the back-edge
  /// comparison): suppression still applies, but the set must not be
  /// rewritten in place -- the head's edges were processed long ago.
  bool SealedEntry = false;

  /// A loop join compares a back edge's exit set against an entry set
  /// analyzed previously: it diagnoses, but must not rewrite that set --
  /// except at a continue latch, a loop-kind join still accumulating a
  /// block's entry set (isUnsealedLoopJoin()).
  bool isLoopJoin() const { return EntryLEK == LEK_LockedSomeLoopIterations; }
  bool canModify() const { return !isLoopJoin(); }
  bool isUnsealedLoopJoin() const { return isLoopJoin() && !SealedEntry; }
  /// Whether the join may keep a stale result as evidence: only the joins
  /// still accumulating a block's entry set -- a branch join, or a continue
  /// latch whose head is not sealed yet. The end-of-function comparison is
  /// not one of them: nothing after it can read the evidence.
  bool mayKeepReleasedTryFact() const {
    return EntryLEK == LEK_LockedSomePredecessors || isUnsealedLoopJoin();
  }
};

/// Class which implements the core thread safety analysis routines.
class ThreadSafetyAnalyzer {
  friend class BuildLockset;
  friend class threadSafety::BeforeSet;
  friend class LocksetJoin;

  llvm::BumpPtrAllocator Bpa;
  threadSafety::til::MemRegionRef Arena;
  threadSafety::SExprBuilder SxBuilder;

  ThreadSafetyHandler &Handler;
  const FunctionDecl *CurrentFunction;
  ASTContext *ASTCtx = nullptr;
  LocalVariableMap LocalVarMap;
  // The beta unchecked-result diagnostics already emitted, keyed by
  // "<join location>:<acquisition location>:<capability>". A join of three
  // or more predecessors is intersected pairwise, and some predecessor
  // orders lose the same try fact twice -- e.g. (conditionally held, not held,
  // conditionally held): the middle predecessor removes it from the entry set
  // with a diagnostic, then the last predecessor re-supplies it one-sided and
  // would diagnose the same leak again (intersectAndWarn()).
  /// The acquisitions already reported as never checked, by originating call,
  /// capability expression and lock kind: one unchecked result is one leak,
  /// however many joins lose the try fact that records it. Keyed on the
  /// capability's expression rather than its printed name, so that two
  /// capabilities that print alike are two leaks.
  llvm::DenseSet<
      std::tuple<const Expr *, const threadSafety::til::SExpr *, unsigned>>
      NeverCheckedWarned;
  // Maps constructed objects to `this` placeholder prior to initialization.
  llvm::SmallDenseMap<const Expr *, til::LiteralPtr *> ConstructedObjects;
  /// The capabilities named by a try-acquire call's attributes, translated
  /// in the call's own context and grouped by the attribute's lock kind and
  /// success value (Falsy: reported acquired when the call returns false).
  struct TryAcquireCaps {
    CapExprSet TruthyExclusive, TruthyShared;
    CapExprSet FalsyExclusive, FalsyShared;
    /// Capabilities whose attribute keys the acquisition to a specific
    /// nonzero integer success code (a constant that is not a bool, e.g.
    /// TRY_ACQUIRE(2, mu)): one entry per attribute per capability.
    /// Falsy values need no entry -- zero is the only falsy integer, so a
    /// falsy attribute's success region is exact already -- and a boolean
    /// success value promises acquisition on any nonzero result
    /// (TruthyAny).
    llvm::SmallVector<std::pair<CapabilityExpr, llvm::APSInt>, 2> ExactCodes;
    /// Capabilities with a truthy success value that is not a specific
    /// integer code (`true`, or a value the constant evaluator cannot
    /// compute): acquired on any nonzero result, so codes recorded for
    /// the same capability by other attributes do not bound its region.
    CapExprSet TruthyAny;
    /// Capabilities reconcileTryAcquireCaps() moved out of the polarity
    /// groups: acquired regardless of the call's result. handleCall()
    /// turns them into unconditional acquisitions, with the diagnostic.
    /// Exclusive only when both polarities promised an exclusive hold; a
    /// cross-kind pairing guarantees no more than a shared hold either
    /// way.
    CapExprSet UnconditionalExclusive, UnconditionalShared;
    /// The capabilities addTryLock() created a try fact for when the walk
    /// last reached the call, each with the lock kind it created it in.
    /// getEdgeLockset() re-materializes a lost hold only for these: a
    /// capability the call names but whose try fact addTryLock() declined
    /// -- diagnosed as untrackable, or acquired definitely by the same
    /// construction -- was never acquired conditionally, so no branch on
    /// the result may manufacture a hold for it. Empty until the walk
    /// reaches the call, which is what keeps a branch decoded above the
    /// call -- a loop-top `if (ok)` over `ok = mu.TryLock()` -- from
    /// re-materializing on the first iteration
    /// (tryheld_retry_with_continue).
    SmallVector<std::pair<CapabilityExpr, LockKind>, 2> TrackedCaps;

    /// Whether the call's attributes name any capability conditionally --
    /// the groups decodeTrylockBranch() builds its per-edge resolutions
    /// from. An unconditional acquisition resolves on no edge, so no try
    /// fact of one exists to re-resolve.
    bool hasConditionalCap() const {
      return !TruthyExclusive.empty() || !TruthyShared.empty() ||
             !FalsyExclusive.empty() || !FalsyShared.empty();
    }

    /// Whether addTryLock() installed a try fact for \p CE in kind \p LK.
    /// One try fact per call and capability, so the kind of the entry is
    /// the kind of the acquisition: an edge cap of any other kind names a
    /// hold this call never took, whichever of the attribute's groups
    /// spelled it.
    bool tracks(const CapabilityExpr &CE, LockKind LK) const {
      return llvm::any_of(TrackedCaps, [&](const auto &Tracked) {
        return Tracked.second == LK && Tracked.first.equals(CE);
      });
    }
  };
  /// The profile of the attributes naming \p Probe among \p Caps.
  static CapProfile getCapProfile(const TryAcquireCaps &Caps,
                                  const CapabilityExpr &Probe);

  // Maps each try-acquire call to its attributes' capabilities, memoized
  // before the lockset walk.
  llvm::SmallDenseMap<const Expr *, TryAcquireCaps> TryAcquireCapsMap;
  FactManager FactMan;
  std::vector<CFGBlockInfo> BlockInfo;

  BeforeSet *GlobalBeforeSet;

public:
  ThreadSafetyAnalyzer(ThreadSafetyHandler &H, BeforeSet *Bset)
      : Arena(&Bpa), SxBuilder(Arena), Handler(H), FactMan(Bpa),
        GlobalBeforeSet(Bset) {}

  bool inCurrentScope(const CapabilityExpr &CapE);

  void addLock(FactSet &FSet, const FactEntry *Entry, bool ReqAttr = false);
  bool addTryLock(FactSet &FSet, const CapabilityExpr &CE, LockKind LK,
                  SourceLocation Loc, const Expr *Call,
                  FactEntry::SourceKind Src = FactEntry::Acquired);
  void checkAcquiredCapability(FactSet &FSet, const FactEntry &Entry,
                               bool ReqAttr);
  bool callResolvesConditionally(const Expr *Call);
  void injectLoopReleasedTryFacts(const CFGBlock *Head, const CFGBlock *Latch,
                                  PostOrderCFGView::CFGBlockSet &Visited);
  void removeLock(FactSet &FSet, const CapabilityExpr &CapE,
                  SourceLocation UnlockLoc, bool FullyRemove, LockKind Kind);

  template <typename AttrType>
  void getMutexIDs(CapExprSet &Mtxs, AttrType *Attr, const Expr *Exp,
                   const NamedDecl *D, til::SExpr *Self = nullptr);

  TryAcquireCaps &recordTryAcquireCall(const Expr *Exp, const NamedDecl *D,
                                       til::SExpr *Self = nullptr,
                                       TryAcquireCaps *NoExprCaps = nullptr);
  TryAcquireCaps *recordedTryAcquireCaps(const Expr *Exp);
  bool sameTryAcquireCaps(const Expr *A, const Expr *B);
  void spendTryAcquiresOf(const CapabilityExpr &Cp);
  void recordTryAcquireCalls();
  void reconcileTryAcquireCaps(const Expr *Exp, TryAcquireCaps &Caps);

  /// What decodeTrylockCond()'s walk over a branched-on condition has learned
  /// about the expressions applied to the try-acquire call's result.
  struct TrylockDecode {
    /// The try-acquire call the walk reached, or null if it found none.
    const CallExpr *TrylockCall = nullptr;
    /// The condition tests the negation of the call's result.
    bool Negate = false;
    /// Set when the branched-on variable merges the call's result with a
    /// constant: the branch-condition truthiness of the edges where the
    /// value may be the constant rather than the call's result (see
    /// decodeTrylockCond()).
    std::optional<bool> AmbiguousCond;
    /// Set when the branched-on value is a `?:` merge that identifies the
    /// call's result on only one truthiness: the branch-condition
    /// truthiness of the edges that determine nothing about the result.
    /// Unlike AmbiguousCond the call did execute on such an edge -- its
    /// result is simply undetermined there -- so the edge must not resolve
    /// any fact (see decodeTrylockCond()).
    std::optional<bool> UnknownCond;
    /// The second of two structurally identical try-acquire calls whose
    /// merged result the condition branches on; null otherwise.
    const CallExpr *MergedCall = nullptr;
    /// Set when the condition compares the (possibly stored) result
    /// against a specific nonzero integer constant: the compared value.
    /// The condition is then `result == CmpValue`, inverted per Negate;
    /// resolveTrylockEdge() resolves the comparison's edges against each
    /// capability's exact success codes.
    std::optional<llvm::APSInt> CmpValue;
    /// Set when the walk to the call passed through a conversion that can
    /// change the value (`bool ok = f();` narrowing an int result). The
    /// condition then tests the converted copy, so a compared value says
    /// nothing about the call's own result and only its truthiness
    /// carries; CmpValue is dropped (decodeTrylockBranch()).
    bool ValueNarrowed = false;
    /// The destination type of the innermost of those conversions -- the
    /// type the value is truncated to, every conversion above it keeping
    /// the value. A constant a merge carries reaches the comparison
    /// converted to it, so that is where constantMeetsCond() has to read
    /// it, and a compared value the type cannot represent is never equal
    /// to the value at all. Null while ValueNarrowed is set means the
    /// walk merged two paths that narrowed differently, where no such
    /// conclusion is available.
    QualType NarrowedTo;
    /// Set when one of those conversions can also make a nonzero result
    /// read as falsy -- a truncation, where `short s = f()` is zero for a
    /// result of 65536. The falsy edge then proves nothing at all, while
    /// the truthy one still proves the result nonzero, since a conversion
    /// of zero is zero (decodeTrylockBranch()).
    bool TruthinessLost = false;
    /// The type of the operand CmpValue is compared against, as written
    /// before the comparison's own promotions. A boolean operand makes
    /// `== 1` a truthiness test and any other value impossible, whatever
    /// the call's own return type is (`_Bool ok = f();`).
    QualType CmpType;
    /// The branched-on value is a `?:` merge of two constants selected by
    /// the result: its truthiness tracks the result's, but its magnitude
    /// is the arm's, so an exact value an edge carries (a case label)
    /// must not be applied to the result (clears
    /// TrylockBranch::ValueIsResult).
    bool ValueMerged = false;
    /// The walk left the block's own terminator condition -- through a
    /// stored variable, a comparison, an assignment, a `?:` value merge,
    /// or a statement expression. The branch block may then be a value
    /// join that a short-circuit edge also reaches, so the `&&`/`||`
    /// descents -- sound only for the block evaluating the operator's
    /// right-hand side in situ -- are refused.
    bool CrossedBlocks = false;
    /// The merged ("phi") definitions the walk has descended into. A
    /// phi-converted loop head makes the definition graph cyclic (`ok2 =
    /// ok; ok = ok2;` inside the loop resolves each variable's merge
    /// through the other's), so a merge reached again resolves nothing
    /// (decodeTrylockCond()).
    llvm::SmallPtrSet<const LocalVariableMap::VarDefinition *, 4> VisitedPhis;
    /// The block whose terminator is being decoded: where the branch reads
    /// the variable, which decides whether an escape has happened yet
    /// (escapeReaches()).
    const CFGBlock *UseBlock = nullptr;
    /// The walk descended into the right-hand side of a `&&` or `||`. The
    /// left-hand side can decide the condition on its own, and when the
    /// block carrying the whole condition is also reached through the
    /// short-circuit -- `while (!(i >= n || ok))`, where the `||`'s value
    /// is materialized for the `!` -- an edge out of it says nothing about
    /// the call on that path.
    bool ShortCircuit = false;

    /// The walk reaches a negation or a comparison below a value comparison
    /// (CmpValue): what is compared is that operator's own boolean, so
    /// `(!x) == 1` and `(x == 0) == 1` are `!x` and `x == 0` themselves --
    /// the comparison is dropped and the walk continues -- and any other
    /// compared value is impossible: returns false, and the branch resolves
    /// nothing. (In C++ the operand's type says as much already, CmpType;
    /// in C both operators yield int.) Drops the comparison in the first
    /// case, so the walk continues on the operator's own terms.
    bool foldCompareToBoolean() {
      if (!CmpValue)
        return true;
      if (!CmpValue->isOne())
        return false;
      CmpValue.reset();
      return true;
    }
  };

  void decodeTrylockCond(const Stmt *Cond, LocalVarContext C, TrylockDecode &D);
  bool escapeReaches(const NamedDecl *VD, const CFGBlock *Use);
  llvm::SmallDenseMap<const CFGBlock *, llvm::BitVector, 2> ReachableFrom;

  // Memoize per-switch summaries (resolveTrylockEdge() visits a switch
  // once per successor).
  llvm::SmallDenseMap<const SwitchStmt *, SwitchSummary, 4> SwitchSummaries;
  const SwitchSummary &getSwitchSummary(ASTContext &Ctx, const SwitchStmt *SW);
  const TrylockBranch &decodeTrylockBranch(const CFGBlock *Block);

  /// The try-acquire calls a block's terminator branches on: the call, and
  /// the second of two identical calls whose merged result it branches on
  /// (TrylockBranch::MergedCall).
  struct TerminatorTrylockCall {
    const CallExpr *TrylockCall = nullptr;
    const CallExpr *MergedCall = nullptr;
  };
  TerminatorTrylockCall getTerminatorTrylockCall(const CFGBlock *Block);
  /// The try-acquire calls a condition branches on, found through its
  /// short-circuit blocks (getConditionTrylockCallExpr()).
  struct ConditionTrylockCall : TerminatorTrylockCall {
    /// Whether every outgoing path of the block reaches the branch on the
    /// call's result; computed on request only.
    bool ResolvesAllPaths = true;
  };
  ConditionTrylockCall getConditionTrylockCallExpr(const CFGBlock *Block,
                                                   bool CheckAllPaths = false);

  /// What one edge proves about the capabilities of a TrylockBranch.
  struct TrylockEdge {
    const CallExpr *TrylockCall = nullptr;
    /// The terminator reached the call through a short-circuit operand
    /// (TrylockBranch::ShortCircuit).
    bool ShortCircuit = false;
    /// The second of two identical calls whose merged result the branch
    /// tests (TrylockBranch::MergedCall): its try facts resolve on this
    /// edge exactly like TrylockCall's.
    const CallExpr *MergedCall = nullptr;
    /// The edge cannot be taken at all (e.g. the implicit default of a
    /// switch that lists every value of a boolean condition).
    bool Infeasible = false;
    /// The call may not have executed on this edge (the branched-on
    /// variable merges its result with a constant): each capability's
    /// resolution holds only if it did.
    bool Ambiguous = false;
    SmallVector<TrylockEdgeCap, 2> Caps;
  };
  TrylockEdge resolveTrylockEdge(const CFGBlock *PredBlock,
                                 const CFGBlock *CurrBlock);

  bool getEdgeLockset(FactSet &Result, const FactSet &ExitSet,
                      const CFGBlock *PredBlock, const CFGBlock *CurrBlock);

  bool join(const FactEntry &A, const FactEntry &B, SourceLocation JoinLoc,
            LockErrorKind EntryLEK);

  void intersectAndWarn(FactSet &EntrySet, const FactSet &ExitSet,
                        const JoinContext &Ctx);

  void intersectAndWarn(FactSet &EntrySet, const FactSet &ExitSet,
                        SourceLocation JoinLoc, LockErrorKind LEK) {
    intersectAndWarn(EntrySet, ExitSet, JoinContext{JoinLoc, LEK, LEK});
  }

  void runAnalysis(AnalysisDeclContext &AC);

  void warnIfMutexNotHeld(const FactSet &FSet, const NamedDecl *D,
                          const Expr *Exp, AccessKind AK, Expr *MutexExp,
                          ProtectedOperationKind POK, til::SExpr *Self,
                          SourceLocation Loc);
  void warnIfAnyMutexNotHeldForRead(const FactSet &FSet, const NamedDecl *D,
                                    const Expr *Exp,
                                    llvm::ArrayRef<Expr *> Args,
                                    ProtectedOperationKind POK,
                                    SourceLocation Loc);
  void warnIfMutexHeld(const FactSet &FSet, const NamedDecl *D, const Expr *Exp,
                       Expr *MutexExp, til::SExpr *Self, SourceLocation Loc);

  void checkAccess(const FactSet &FSet, const Expr *Exp, AccessKind AK,
                   ProtectedOperationKind POK);
  void checkPtAccess(const FactSet &FSet, const Expr *Exp, AccessKind AK,
                     ProtectedOperationKind POK);
};

} // namespace

/// Process acquired_before and acquired_after attributes on Vd.
BeforeSet::BeforeInfo* BeforeSet::insertAttrExprs(const ValueDecl* Vd,
    ThreadSafetyAnalyzer& Analyzer) {
  // Create a new entry for Vd.
  BeforeInfo *Info = nullptr;
  {
    // Keep InfoPtr in its own scope in case BMap is modified later and the
    // reference becomes invalid.
    std::unique_ptr<BeforeInfo> &InfoPtr = BMap[Vd];
    if (!InfoPtr)
      InfoPtr.reset(new BeforeInfo());
    Info = InfoPtr.get();
  }

  for (const auto *At : Vd->attrs()) {
    switch (At->getKind()) {
      case attr::AcquiredBefore: {
        const auto *A = cast<AcquiredBeforeAttr>(At);

        // Read exprs from the attribute, and add them to BeforeVect.
        for (const auto *Arg : A->args()) {
          CapabilityExpr Cp =
            Analyzer.SxBuilder.translateAttrExpr(Arg, nullptr);
          if (const ValueDecl *Cpvd = Cp.valueDecl()) {
            Info->Vect.push_back(Cpvd);
            const auto It = BMap.find(Cpvd);
            if (It == BMap.end())
              insertAttrExprs(Cpvd, Analyzer);
          }
        }
        break;
      }
      case attr::AcquiredAfter: {
        const auto *A = cast<AcquiredAfterAttr>(At);

        // Read exprs from the attribute, and add them to BeforeVect.
        for (const auto *Arg : A->args()) {
          CapabilityExpr Cp =
            Analyzer.SxBuilder.translateAttrExpr(Arg, nullptr);
          if (const ValueDecl *ArgVd = Cp.valueDecl()) {
            // Get entry for mutex listed in attribute
            BeforeInfo *ArgInfo = getBeforeInfoForDecl(ArgVd, Analyzer);
            ArgInfo->Vect.push_back(Vd);
          }
        }
        break;
      }
      default:
        break;
    }
  }

  return Info;
}

BeforeSet::BeforeInfo *
BeforeSet::getBeforeInfoForDecl(const ValueDecl *Vd,
                                ThreadSafetyAnalyzer &Analyzer) {
  auto It = BMap.find(Vd);
  BeforeInfo *Info = nullptr;
  if (It == BMap.end())
    Info = insertAttrExprs(Vd, Analyzer);
  else
    Info = It->second.get();
  assert(Info && "BMap contained nullptr?");
  return Info;
}

/// Return true if any mutexes in FSet are in the acquired_before set of Vd.
void BeforeSet::checkBeforeAfter(const ValueDecl* StartVd,
                                 const FactSet& FSet,
                                 ThreadSafetyAnalyzer& Analyzer,
                                 SourceLocation Loc, StringRef CapKind) {
  SmallVector<BeforeInfo*, 8> InfoVect;

  // Do a depth-first traversal of Vd.
  // Return true if there are cycles.
  std::function<bool (const ValueDecl*)> traverse = [&](const ValueDecl* Vd) {
    if (!Vd)
      return false;

    BeforeSet::BeforeInfo *Info = getBeforeInfoForDecl(Vd, Analyzer);

    if (Info->Visited == 1)
      return true;

    if (Info->Visited == 2)
      return false;

    if (Info->Vect.empty())
      return false;

    InfoVect.push_back(Info);
    Info->Visited = 1;
    for (const auto *Vdb : Info->Vect) {
      // Exclude mutexes in our immediate before set.
      if (FSet.containsMutexDecl(Analyzer.FactMan, Vdb)) {
        StringRef L1 = StartVd->getName();
        StringRef L2 = Vdb->getName();
        Analyzer.Handler.handleLockAcquiredBefore(CapKind, L1, L2, Loc);
      }
      // Transitively search other before sets, and warn on cycles.
      if (traverse(Vdb)) {
        if (CycMap.try_emplace(Vd, true).second) {
          StringRef L1 = Vd->getName();
          Analyzer.Handler.handleBeforeAfterCycle(L1, Vd->getLocation());
        }
      }
    }
    Info->Visited = 2;
    return false;
  };

  traverse(StartVd);

  for (auto *Info : InfoVect)
    Info->Visited = 0;
}

/// Gets the value decl pointer from DeclRefExprs or MemberExprs.
static const ValueDecl *getValueDecl(const Expr *Exp) {
  if (const auto *CE = dyn_cast<ImplicitCastExpr>(Exp))
    return getValueDecl(CE->getSubExpr());

  if (const auto *DR = dyn_cast<DeclRefExpr>(Exp))
    return DR->getDecl();

  if (const auto *ME = dyn_cast<MemberExpr>(Exp))
    return ME->getMemberDecl();

  return nullptr;
}

bool ThreadSafetyAnalyzer::inCurrentScope(const CapabilityExpr &CapE) {
  const threadSafety::til::SExpr *SExp = CapE.sexpr();
  assert(SExp && "Null expressions should be ignored");

  if (const auto *LP = dyn_cast<til::LiteralPtr>(SExp)) {
    const ValueDecl *VD = LP->clangDecl();
    // Variables defined in a function are always inaccessible.
    if (!VD || !VD->isDefinedOutsideFunctionOrMethod())
      return false;
    // For now we consider static class members to be inaccessible.
    if (isa<CXXRecordDecl>(VD->getDeclContext()))
      return false;
    // Global variables are always in scope.
    return true;
  }

  // Members are in scope from methods of the same class.
  if (const auto *P = dyn_cast<til::Project>(SExp)) {
    if (!isa_and_nonnull<CXXMethodDecl>(CurrentFunction))
      return false;
    const ValueDecl *VD = P->clangDecl();
    return VD->getDeclContext() == CurrentFunction->getDeclContext();
  }

  return false;
}

/// Add a new lock to the lockset, warning if the lock is already there.
/// \param ReqAttr -- true if this is part of an initial Requires attribute.
void ThreadSafetyAnalyzer::addLock(FactSet &FSet, const FactEntry *Entry,
                                   bool ReqAttr) {
  if (Entry->shouldIgnore())
    return;
  assert(!isa<TryFactEntry>(Entry) && "try facts are added by addTryLock");

  checkAcquiredCapability(FSet, *Entry, ReqAttr);

  // The capability is acquired here, so the record of a hold released
  // earlier no longer describes anything a branch could resurrect: a stale
  // result kept past the re-acquisition would make a release inside a loop
  // that re-acquires look, at the loop's exit, like a release the loop left
  // standing (injectLoopReleasedTryFacts()). An assert claims a hold rather
  // than taking one, and leaves the record alone.
  if (!Entry->asserted())
    FSet.retireReleased(FactMan, *Entry);

  if (const FactEntry *Cp = FSet.findDefinite(FactMan, *Entry)) {
    // Held already: reacquire reentrantly or diagnose it.
    // Conditional try facts beside the definite fact are unaffected -- each
    // is resolved by the branch on its own result.
    if (!Entry->asserted())
      Cp->handleLock(FSet, FactMan, *Entry, Handler);
    return;
  }
  if (const FactEntry *Cond = FSet.firstConditional(FactMan, *Entry)) {
    if (Entry->asserted()) {
      // An assert directly upgrades the capability to held, without a
      // diagnostic: it claims exactly that knowledge, and the conditional
      // acquisitions are subsumed by it -- their results, once checked,
      // must not add levels the assert already accounts for.
      FSet.removeAllConditional(FactMan, *Entry);
      FSet.addLock(FactMan, Entry);
      return;
    }
    addLockOverConditional(FSet, FactMan, Entry, *Cond, &Handler);
    return;
  }
  FSet.addLock(FactMan, Entry);
}

/// The checks an acquisition performs: consume (or require) the negative
/// capability, and check acquired_before/acquired_after ordering. A
/// try-acquire attempts the acquisition, so a try fact \p Entry is checked
/// the same way -- once, at the call. The negative capability is consumed
/// either way: after the call the capability is possibly held, so a
/// negative fact that predates it no longer describes the state; on the
/// call's failure edge getEdgeLockset() re-establishes it beside the call's
/// ProvedNotHeld try fact.
void ThreadSafetyAnalyzer::checkAcquiredCapability(FactSet &FSet,
                                                   const FactEntry &Entry,
                                                   bool ReqAttr) {
  if (!ReqAttr && !Entry.negative()) {
    // The acquisition consumes the negative fact -- what its negative
    // capability requirement asks for. The try facts of the capability are
    // untouched: a stale or failed stored result stays stale or failed
    // across an acquisition by another call.
    CapabilityExpr NegC = !Entry;
    if (!FSet.removeDefinite(FactMan, NegC) && inCurrentScope(Entry) &&
        !Entry.asserted() && !Entry.reentrant())
      Handler.handleNegativeNotHeld(Entry.getKind(), Entry.toString(),
                                    NegC.toString(), Entry.loc());
  }

  // Check before/after constraints
  if (!Entry.asserted() && !Entry.declared()) {
    GlobalBeforeSet->checkBeforeAfter(Entry.valueDecl(), FSet, *this,
                                      Entry.loc(), Entry.getKind());
  }
}

/// Whether branching on \p Call's result can resolve any capability at
/// all. A call whose attributes all reconciled to unconditional
/// acquisitions (TRY_ACQUIRE(true, mu) TRY_ACQUIRE(false, mu)) names none
/// conditionally, so its branch decodes to nothing -- letting it claim a
/// `?:` condition would only discard an arm that does resolve.
bool ThreadSafetyAnalyzer::callResolvesConditionally(const Expr *Call) {
  auto It = TryAcquireCapsMap.find(Call);
  return It != TryAcquireCapsMap.end() && It->second.hasConditionalCap();
}

/// Add a conditional try fact of the try-acquire call \p Call at \p Loc for
/// the capability \p CE. It joins the capability's other facts: a definite
/// hold, which the success edge deepens (at runtime a try-acquire over a
/// held capability fails rather than deadlocks, and even a reentrant one
/// may fail), and other calls' try facts, each resolved by its own branch.
/// A resolved try fact of this same call starts over: a fresh execution
/// overwrites the stored result. What cannot be tracked is diagnosed and
/// left untracked: a hold of the other kind (shared vs. exclusive), one
/// this acquisition can neither nest in nor coexist with, and a repeat of
/// this call over its own unresolved try fact (one try fact per call and
/// kind). A hold of a scoped object is not such a case: the call is one of
/// the guard's own members, and speaks about what the guard manages.
/// \p Src is Managed for a scoped lockable's construction, whose destructor
/// conditionally releases (disarms) the try fact. Whether a try fact was
/// installed, so that a scoped object manages what it really acquired.
bool ThreadSafetyAnalyzer::addTryLock(FactSet &FSet, const CapabilityExpr &CE,
                                      LockKind LK, SourceLocation Loc,
                                      const Expr *Call,
                                      FactEntry::SourceKind Src) {
  auto *Fact = FactMan.createFact<TryFactEntry>(CE, LK, Loc, Src, Call);
  if (Fact->shouldIgnore())
    return false;

  checkAcquiredCapability(FSet, *Fact, /*ReqAttr=*/false);

  if (const FactEntry *Cp = FSet.findDefinite(FactMan, CE)) {
    // A try-acquire naming a scoped object -- a guard's own TryLock() member --
    // is trackable: the try fact speaks about the guard, and its success edge
    // replays what the guard acquires, as a blocking member of the guard does.
    if (!isa<ScopedLockableFactEntry>(Cp) && Cp->kind() != LK) {
      Handler.handleDoubleLock(CE.getKind(), CE.toString(), Cp->loc(), Loc,
                               /*MaybeHeld=*/false);
      return false;
    }
  }
  SmallVector<const TryFactEntry *, 2> TryFacts;
  FSet.collectTryFacts(FactMan, CE, TryFacts);
  for (const TryFactEntry *W : TryFacts) {
    if (W->conditional()) {
      if (W->kind() != LK || W->origin() == Call) {
        Handler.handleDoubleLock(CE.getKind(), CE.toString(), W->loc(), Loc,
                                 /*MaybeHeld=*/true);
        return false;
      }
      // Another call's result is still pending here, so both calls may end
      // up holding the capability: neither result stands for the other, and
      // a merge of the two is not the retry idiom
      // (FactManager::coexecutedTryAcquire()).
      FactMan.addCoexecutedTryAcquire(W->origin());
      FactMan.addCoexecutedTryAcquire(Call);
    } else if (W->origin() == Call && W->kind() == LK) {
      // A fresh execution of the call overwrites its stored result: a hold
      // an earlier execution proved is no longer determined by it, and the
      // try fact starts over as the new result's.
      FSet.removeFact(FactMan, *W);
    }
  }
  FSet.addLock(FactMan, Fact);
  return true;
}

/// Remove a lock from the lockset, warning if the lock is not there.
/// \param UnlockLoc The source location of the unlock (only used in error msg)
void ThreadSafetyAnalyzer::removeLock(FactSet &FSet, const CapabilityExpr &Cp,
                                      SourceLocation UnlockLoc,
                                      bool FullyRemove, LockKind ReceivedKind) {
  if (Cp.shouldIgnore())
    return;

  const FactEntry *LDat = FSet.findDefinite(FactMan, Cp);
  if (!LDat) {
    // The release is unconditional and the capability is not definitely
    // held: it took a hold or aborted, so nothing is held afterwards and
    // every stored result for the capability is stale. Conditional try
    // facts record that themselves (releaseConditional() below); a result
    // whose try fact an earlier join already lost has no fact left to
    // record it, so the call is spent instead -- otherwise a later branch
    // on it re-materializes the hold this release gave up.
    spendTryAcquiresOf(Cp);
    if (handleUncheckedConditionalUnlock(FSet, FactMan, Cp, UnlockLoc,
                                         &Handler))
      return;
    Handler.handleUnmatchedUnlock(Cp.getKind(), Cp.toString(), UnlockLoc,
                                  unmatchedUnlockNoteLoc(FSet, FactMan, Cp),
                                  false);
    return;
  }

  // Generic lock removal doesn't care about lock kind mismatches, but
  // otherwise diagnose when the lock kinds are mismatched.
  if (ReceivedKind != LK_Generic && LDat->kind() != ReceivedKind) {
    Handler.handleIncorrectUnlockKind(Cp.getKind(), Cp.toString(), LDat->kind(),
                                      ReceivedKind, LDat->loc(), UnlockLoc);
  }

  LDat->handleUnlock(FSet, FactMan, Cp, UnlockLoc, FullyRemove, Handler);
}

/// Extract the list of mutexIDs from the attribute on an expression,
/// and push them onto Mtxs, discarding any duplicates.
template <typename AttrType>
void ThreadSafetyAnalyzer::getMutexIDs(CapExprSet &Mtxs, AttrType *Attr,
                                       const Expr *Exp, const NamedDecl *D,
                                       til::SExpr *Self) {
  if (Attr->args_size() == 0) {
    // The mutex held is the "this" object.
    CapabilityExpr Cp = SxBuilder.translateAttrExpr(nullptr, D, Exp, Self);
    if (Cp.isInvalid()) {
      warnInvalidLock(Handler, nullptr, D, Exp, Cp.getKind());
      return;
    }
    //else
    if (!Cp.shouldIgnore())
      Mtxs.push_back_nodup(Cp);
    return;
  }

  for (const auto *Arg : Attr->args()) {
    CapabilityExpr Cp = SxBuilder.translateAttrExpr(Arg, D, Exp, Self);
    if (Cp.isInvalid()) {
      warnInvalidLock(Handler, nullptr, D, Exp, Cp.getKind());
      continue;
    }
    //else
    if (!Cp.shouldIgnore())
      Mtxs.push_back_nodup(Cp);
  }
}

// Returns whether E is a compile-time constant, setting TCond to its boolean
// value. Looks through parentheses and evaluates constant expressions
// (constexpr values, enumerators), not just literals.
static bool getStaticBooleanValue(const Expr *E, bool &TCond,
                                  const ASTContext &Ctx) {
  return !E->isValueDependent() && E->EvaluateAsBooleanCondition(TCond, Ctx);
}

/// Whether \p VD's storage may already have escaped when \p Use's terminator
/// reads it: some block where it escapes reaches \p Use, itself included, so
/// on some path a reference to the variable is live before the branch. An
/// escape only reachable *after* the branch -- an address taken past the
/// checked region -- leaves the branch alone. Reachability is memoized per
/// escaping block; a function with no escaped variable computes none.
bool ThreadSafetyAnalyzer::escapeReaches(const NamedDecl *VD,
                                         const CFGBlock *Use) {
  ArrayRef<const CFGBlock *> Escapes = LocalVarMap.escapedAt(VD);
  if (Escapes.empty() || !Use)
    return false;
  for (const CFGBlock *E : Escapes) {
    if (!E)
      return true; // Recorded without a block: assume the worst.
    auto [It, Inserted] = ReachableFrom.try_emplace(E);
    llvm::BitVector &Reached = It->second;
    if (Inserted) {
      // The walk stops at the block declaring the variable: re-entering the
      // declaration is fresh storage, which the reference that escaped the
      // old storage cannot reach into. That is what keeps a result declared
      // inside a loop resolvable when the escape sits at the end of the
      // body -- the only path back to the branch runs through the
      // declaration.
      const CFGBlock *Decl = LocalVarMap.declaredIn(VD);
      Reached.resize(BlockInfo.size());
      SmallVector<const CFGBlock *, 16> Work{E};
      Reached.set(E->getBlockID());
      while (!Work.empty()) {
        const CFGBlock *B = Work.pop_back_val();
        for (const CFGBlock *S : B->succs())
          if (S && S != Decl && !Reached.test(S->getBlockID())) {
            Reached.set(S->getBlockID());
            Work.push_back(S);
          }
      }
    }
    if (Reached.test(Use->getBlockID()))
      return true;
  }
  return false;
}

// Whether \p CE leaves every value of its operand intact, so that a
// comparison of its result against a constant is a comparison of the
// operand against that same constant. True for the integral promotions and
// widenings a condition applies on the way to the branch (an enum or a
// `short` compared as an `int`, the `long` inside __builtin_expect); false
// for a conversion to `bool`, a narrowing one, or anything not plainly
// integral, where the compared copy no longer identifies the operand.
static bool castPreservesValue(const ASTContext &Ctx,
                               const ImplicitCastExpr *CE) {
  QualType To = CE->getType(), From = CE->getSubExpr()->getType();
  if (Ctx.hasSameUnqualifiedType(To, From))
    return true;
  switch (CE->getCastKind()) {
  case CK_LValueToRValue:
  case CK_NoOp:
    return true;
  case CK_IntegralCast:
    break;
  default:
    return false;
  }
  if (To->isBooleanType() || !To->isIntegralOrEnumerationType() ||
      !From->isIntegralOrEnumerationType())
    return false;
  // An enumeration's values are its enumerators', not its underlying
  // type's: without a fixed underlying type its promotion type represents
  // them all by definition (`enum { kFailed = 0, kAcquired = 5 }` is
  // unsigned underneath, yet promotes to int without loss), while a fixed
  // underlying type is the enumeration's whole range.
  if (const EnumDecl *ED = From->getAsEnumDecl()) {
    From = ED->isFixed() ? ED->getIntegerType() : ED->getPromotionType();
    if (From.isNull())
      return false;
    if (Ctx.hasSameUnqualifiedType(To, From))
      return true;
  }
  // The destination must represent every value of the source: strictly
  // wider keeps them all unless a signed source meets an unsigned
  // destination, and equal width only when the signedness agrees.
  const unsigned ToBits = Ctx.getIntWidth(To), FromBits = Ctx.getIntWidth(From);
  const bool ToSigned = To->isSignedIntegerOrEnumerationType();
  const bool FromSigned = From->isSignedIntegerOrEnumerationType();
  if (ToBits == FromBits)
    return ToSigned == FromSigned;
  return ToBits > FromBits && (ToSigned || !FromSigned);
}

// Whether \p CE maps every nonzero operand to a nonzero result, so that the
// truthiness of what it produces is the operand's own. True for a conversion
// to bool and for every conversion that keeps the value; false for a
// truncation, which can make a nonzero result read as falsy (`short s = f()`
// is zero for a result of 65536), and for anything not plainly integral.
static bool castPreservesTruthiness(const ASTContext &Ctx,
                                    const ImplicitCastExpr *CE) {
  QualType To = CE->getType(), From = CE->getSubExpr()->getType();
  if (To->isBooleanType() || castPreservesValue(Ctx, CE))
    return true;
  if (!To->isIntegralOrEnumerationType() ||
      !From->isIntegralOrEnumerationType())
    return false;
  return Ctx.getIntWidth(To) >= Ctx.getIntWidth(From);
}

// Whether \p V is a value of type \p Ty: converting it there and back
// keeps it. A comparison against a value the operand's type cannot hold is
// never true.
static bool representableIn(const ASTContext &Ctx, const llvm::APSInt &V,
                            QualType Ty) {
  llvm::APSInt Conv = V.extOrTrunc(Ctx.getIntWidth(Ty));
  Conv.setIsSigned(Ty->isSignedIntegerOrEnumerationType());
  return llvm::APSInt::isSameValue(Conv, V);
}

// The truthiness of a constant merge arm as the branch reads it. A
// conversion the walk looked through changes it only where it can make a
// nonzero value falsy (TruthinessLost: `char c = ok ? 256 : 0` stores zero
// whichever arm ran); anything else -- a conversion to bool above all --
// leaves the arm's own truthiness the answer. False where the arm is not a
// constant, or where a truncating conversion's own type is not known.
static bool getMergedArmTruthiness(const Expr *E, bool &Cond,
                                   const ThreadSafetyAnalyzer::TrylockDecode &D,
                                   const ASTContext &Ctx) {
  if (!D.ValueNarrowed || !D.TruthinessLost)
    return getStaticBooleanValue(E, Cond, Ctx);
  if (D.NarrowedTo.isNull() || !D.NarrowedTo->isIntegralOrEnumerationType())
    return false;
  Expr::EvalResult ER;
  if (E->isValueDependent() || !E->EvaluateAsInt(ER, Ctx))
    return false;
  Cond = ER.Val.getInt().extOrTrunc(Ctx.getIntWidth(D.NarrowedTo)) != 0;
  return true;
}

// Whether the condition decoded so far is satisfied when the merged value
// is \p ConstE rather than the call's result, \p K being the constant's
// truthiness. A truthiness branch is satisfied exactly when the constant is
// truthy; a comparison against an exact value, when the constant is that
// value -- as the comparison reads it, which is the constant converted
// through the truncation the walk looked through (TrylockDecode::NarrowedTo:
// `short s = ok ? 65537 : 0` compares s against 1, not 65537). Returns
// nullopt when a compared constant does not evaluate as an integer, or
// where the truncation it went through is not a single known type, and the
// caller can conclude nothing about either edge. The result is the
// condition's truthiness before \c Negate is applied.
static std::optional<bool>
constantMeetsCond(const Expr *ConstE, bool K,
                  const ThreadSafetyAnalyzer::TrylockDecode &D,
                  const ASTContext &Ctx) {
  if (!D.CmpValue)
    return K;
  Expr::EvalResult ER;
  if (!ConstE || ConstE->isValueDependent() || !ConstE->EvaluateAsInt(ER, Ctx))
    return std::nullopt;
  llvm::APSInt Const = ER.Val.getInt();
  if (D.ValueNarrowed) {
    if (D.NarrowedTo.isNull() || !D.NarrowedTo->isIntegralOrEnumerationType())
      return std::nullopt;
    Const = Const.extOrTrunc(Ctx.getIntWidth(D.NarrowedTo));
    Const.setIsSigned(D.NarrowedTo->isSignedIntegerOrEnumerationType());
  }
  return llvm::APSInt::isSameValue(Const, *D.CmpValue);
}

// If Cond can be traced back to a try-acquire function call, the `D` variable
// will be populated with the call and with how the branched-on value relates
// to its result -- negation (e.g. `if (!mu.tryLock(...))`), a comparison
// against a constant, a merge with a constant (directly or through `?:`
// arms), or a merge of two structurally identical calls.
void ThreadSafetyAnalyzer::decodeTrylockCond(const Stmt *Cond,
                                             LocalVarContext C,
                                             TrylockDecode &D) {
  // Continue the decode into one operand of a value merge, under the marks
  // \p Probe carries for the edges that operand cannot account for, and
  // adopt the result only if it found a call this branch can resolve.
  // Whichever operand answers first is the one the branch names: a merge
  // carries one call's result, not two. Leaving the walk (the operand is
  // evaluated in a block of its own) is what CrossedBlocks records.
  auto ProbeOperand = [&](const Expr *E, TrylockDecode Probe,
                          bool MustResolve) {
    Probe.CrossedBlocks = true;
    decodeTrylockCond(E, C, Probe);
    if (!Probe.TrylockCall ||
        (MustResolve && !callResolvesConditionally(Probe.TrylockCall)))
      return false;
    D = Probe;
    return true;
  };

  if (!Cond)
    return;

  if (const auto *CallExp = dyn_cast<CallExpr>(Cond)) {
    if (CallExp->getBuiltinCallee() == Builtin::BI__builtin_expect) {
      // The argument value is materialized for the call, so a logical
      // operator inside it was evaluated across a join, not in situ.
      D.CrossedBlocks = true;
      return decodeTrylockCond(CallExp->getArg(0), C, D);
    }
    // Any other call ends the walk with an empty decode -- e.g. letting a
    // `?:` arm that is not the try-acquire fall back to the condition.
    const auto *FD = dyn_cast_or_null<NamedDecl>(CallExp->getCalleeDecl());
    if (FD && FD->hasAttr<TryAcquireCapabilityAttr>())
      D.TrylockCall = CallExp;
    return;
  }
  else if (const auto *PE = dyn_cast<ParenExpr>(Cond))
    return decodeTrylockCond(PE->getSubExpr(), C, D);
  else if (const auto *CE = dyn_cast<ImplicitCastExpr>(Cond)) {
    // Looking through a conversion that can change the value leaves the
    // condition testing a converted copy of the result, not the result:
    // `bool ok = f(); if (ok == 1)` means "f() was nonzero", so no exact
    // code may be pinned against it. Truthiness survives every such
    // conversion, and is what the branch is left resolving. A conversion
    // whose destination represents every value of its source changes
    // nothing -- the integral promotions a comparison applies to an enum
    // or a `short`, and the widening to `long` inside __builtin_expect,
    // must not cost the exact resolution.
    if (!castPreservesValue(*ASTCtx, CE)) {
      D.ValueNarrowed = true;
      D.NarrowedTo = CE->getType();
      if (!castPreservesTruthiness(*ASTCtx, CE))
        D.TruthinessLost = true;
    }
    return decodeTrylockCond(CE->getSubExpr(), C, D);
  } else if (const auto *FE = dyn_cast<FullExpr>(Cond))
    return decodeTrylockCond(FE->getSubExpr(), C, D);
  else if (const auto *DRE = dyn_cast<DeclRefExpr>(Cond)) {
    // The reasoning below assumes every assignment to the variable is
    // visible in the map. A variable whose reference has escaped (captured
    // or bound by reference, address taken) can be mutated by any call in
    // between, so neither its direct definitions nor its merges identify
    // the branched-on value -- but only from the escape onwards. An escape
    // that cannot reach this branch has not happened yet on any path to it,
    // and the branch reads exactly what the map says (escapeReaches()).
    if (escapeReaches(DRE->getDecl(), D.UseBlock))
      return;
    // Every descent below leaves the terminator condition for a stored
    // definition evaluated in an earlier block.
    D.CrossedBlocks = true;
    LocalVarContext DefCtx = C;
    if (const Expr *E = LocalVarMap.lookupExpr(DRE->getDecl(), DefCtx))
      return decodeTrylockCond(E, DefCtx, D);
    // A merged ("phi") definition: if the variable merges one non-constant
    // definition with a constant of truthiness K (e.g. a try-acquire result
    // stored over a constant initializer), a branch on the variable still
    // identifies the non-constant definition -- on an edge where the
    // variable's truthiness is !K the value can only be that definition's
    // result. Record in AmbiguousCond the branch-condition truthiness of
    // the edges where the value may instead be the constant;
    // getEdgeLockset() refuses to treat those edges as proof that the call
    // executed. Only one merge can be resolved per condition.
    // The merge may sit behind a chain of references (a loop head wraps
    // every variable in a reference definition), so test the canonical
    // definition, not the immediate one.
    const auto *VDef = LocalVarMap.lookupCanonical(DRE->getDecl(), C);
    if (!VDef || !VDef->isPhi() || D.AmbiguousCond)
      return;
    // A merge the walk is already resolving through: the chain is cyclic
    // (a loop-carried copy between two variables), and nothing below can
    // identify the value.
    if (!D.VisitedPhis.insert(VDef).second)
      return;
    ASTContext &ACtx = DRE->getDecl()->getASTContext();
    const Expr *NonConst = nullptr, *NonConst2 = nullptr, *ConstE = nullptr;
    LocalVarContext NonConstCtx = C, NonConstCtx2 = C;
    unsigned NonConstID = 0, ConstID = 0;
    std::optional<bool> K;
    for (unsigned Op : {VDef->DirectRef, VDef->PhiAlt}) {
      LocalVarContext OpCtx = C;
      const Expr *E = LocalVarMap.lookupExprByID(Op, OpCtx);
      if (!E)
        return;
      // Any expression that constant-evaluates counts as the constant, not
      // just a literal: `bool b = kFalseConstant;` merges the same way as
      // `bool b = false;`.
      bool B;
      if (getStaticBooleanValue(E, B, ACtx)) {
        if (K && *K != B)
          return; // Constants of both truthinesses determine nothing.
        K = B;
        ConstID = Op;
        ConstE = E;
      } else if (NonConst) {
        NonConst2 = E;
        NonConstCtx2 = OpCtx;
      } else {
        NonConst = E;
        NonConstCtx = OpCtx;
        NonConstID = Op;
      }
    }
    if (NonConst2) {
      // Two non-constant definitions: a branch still resolves the merge if
      // both are the same branch-relevant expression -- in practice two
      // structurally identical try-acquire calls, as in the retry idiom
      // `ok = mu.TryLock(); while (!ok) ok = mu.TryLock();` -- since either
      // way the variable holds "the result of that call". Resolve to the
      // first path's call: its try fact is the one in the entry set wherever
      // this merge is branched on, and joins have verified the two paths'
      // states agree.
      llvm::FoldingSetNodeID ID1, ID2;
      NonConst->IgnoreParens()->Profile(ID1, ACtx, /*Canonical=*/true);
      NonConst2->IgnoreParens()->Profile(ID2, ACtx, /*Canonical=*/true);
      if (ID1 != ID2)
        return;
      // (In the unsound retry-without-checking variant
      // `b = mu.TryLock(); while (work()) b = mu.TryLock();` the second
      // call executes while the first result may still be pending: its
      // try fact is a second one of the capability, and the first's goes
      // unchecked, which the loop join reports under -Wthread-safety-beta.)
      // The second path's call resolves the same way; report it through
      // MergedCall so a join can fold the two calls' try facts into the
      // resolved call's (intersectAndWarn()).
      //
      // Whenever the second path does not agree, the whole merge is
      // refused, not just the companion: the branch is decoded as a branch
      // on the first path's call, which only holds if the second path
      // reaches the same call in the same sense. Restoring the decode to
      // what it was on entry (BeforeD) leaves the merge unresolved, the
      // conservative answer -- dropping the companion alone would keep an
      // unsound resolution whose outcome depends on predecessor order.
      // Note the expression comparison above does not establish agreement:
      // the same expression (a reference to one variable) can resolve
      // differently per path (`if (c) { t = !t; ok = t; } else ok = t;`).
      //
      // MergedCall cannot be set on entry: its only non-null write below
      // returns without descending further.
      assert(!D.MergedCall && "two-call merge state leaked into a descent");
      const TrylockDecode BeforeD = D;
      decodeTrylockCond(NonConst, NonConstCtx, D);
      const CallExpr *First = D.TrylockCall;
      // Either the first path resolves to no call at all, or its
      // resolution nests a two-call merge of its own, which a single
      // companion call cannot represent.
      if (!First || D.MergedCall) {
        D = BeforeD;
        return;
      }
      TrylockDecode D2 = BeforeD;
      decodeTrylockCond(NonConst2, NonConstCtx2, D2);
      const CallExpr *Second = D2.TrylockCall;
      auto SameCmp = [](const std::optional<llvm::APSInt> &A,
                        const std::optional<llvm::APSInt> &B) {
        return A.has_value() == B.has_value() &&
               (!A || llvm::APSInt::isSameValue(*A, *B));
      };
      if (!Second || D2.MergedCall || D2.Negate != D.Negate ||
          D2.AmbiguousCond != D.AmbiguousCond ||
          D2.UnknownCond != D.UnknownCond || D2.ValueMerged != D.ValueMerged ||
          !SameCmp(D2.CmpValue, D.CmpValue) || D2.CmpType != D.CmpType) {
        D = BeforeD;
        return;
      }
      // A conversion that can change the value on either path leaves the
      // merged value a converted copy on that path, so no code may be
      // pinned against it: the paths' marks are unioned rather than
      // decided by whichever was walked first. Paths that truncate to
      // different types (or only one of which truncates) leave no single
      // type the merged value is a copy in, which NarrowedTo records as
      // null.
      if (D2.ValueNarrowed != D.ValueNarrowed || D2.NarrowedTo != D.NarrowedTo)
        D.NarrowedTo = QualType();
      D.ValueNarrowed |= D2.ValueNarrowed;
      // Both paths reaching the very same call needs no companion: the
      // merged value is that one call's result either way.
      if (Second == First)
        return;
      // A call whose result a join has already lost unchecked cannot be
      // half of a twin: the retry idiom is sound because each call runs
      // only after the previous result was checked false, and a lost
      // result is exactly the case where it was not
      // (`b = mu.TryLock(); while (work()) b = mu.TryLock();`, where the
      // loop re-runs the call over a hold the first call may still have).
      // Refusing the merge leaves the branch unresolved, so the hold is
      // diagnosed in the default group rather than only by the beta report
      // at the loop join.
      if (FactMan.lostUnchecked(First) || FactMan.lostUnchecked(Second) ||
          FactMan.coexecutedTryAcquire(First) ||
          FactMan.coexecutedTryAcquire(Second)) {
        D = BeforeD;
        return;
      }
      // The stored expressions were compared above, but they may be hops
      // (a copy through another variable) that resolved to calls of their
      // own: the identical-resolution premise holds for the calls
      // themselves, so compare those -- and compare what they acquire, not
      // how they read (sameTryAcquireCaps()).
      if (!sameTryAcquireCaps(First, Second)) {
        D = BeforeD;
        return;
      }
      D.MergedCall = Second;
      return;
    }
    if (!NonConst || !K)
      return;
    // The reasoning below is only sound if the constant is not a later
    // overwrite of the non-constant definition (`b = try_lock(); b = false;`
    // -- the capability may be held although the variable is false again):
    // the constant's definition chain must show the non-constant assignment
    // never executed on its paths.
    if (!LocalVarMap.chainAvoids(DRE->getDecl(), ConstID, NonConstID))
      return;
    // On the ambiguous edges the variable's value may be the constant
    // rather than the call's result. Which edge that is depends on what
    // the condition asks: for a truthiness branch it is the edge matching
    // the constant's truthiness K, but against an exact comparison it is
    // the edge the constant itself satisfies -- `(c ? r : 2) == 2` is
    // ambiguous on its true edge, while a constant the comparison rejects
    // leaves that edge pinned to the result and makes the other one
    // ambiguous. Either way the condition's truthiness is adjusted by the
    // negations applied so far.
    std::optional<bool> ConstMeetsCond = constantMeetsCond(ConstE, *K, D, ACtx);
    if (!ConstMeetsCond)
      return; // The constant does not compare: resolve nothing.
    D.AmbiguousCond = *ConstMeetsCond != D.Negate;
    return decodeTrylockCond(NonConst, NonConstCtx, D);
  }
  else if (const auto *UOP = dyn_cast<UnaryOperator>(Cond)) {
    if (UOP->getOpcode() == UO_LNot) {
      if (!D.foldCompareToBoolean())
        return;
      D.Negate = !D.Negate;
      return decodeTrylockCond(UOP->getSubExpr(), C, D);
    }
    return;
  }
  else if (const auto *BOP = dyn_cast<BinaryOperator>(Cond)) {
    if (BOP->getOpcode() == BO_EQ || BOP->getOpcode() == BO_NE) {
      if (!D.foldCompareToBoolean())
        return;
      if (BOP->getOpcode() == BO_NE)
        D.Negate = !D.Negate;

      // Comparison against a constant. A falsy constant inverts the
      // condition (`x == 0` is `!x`) and a truthy bool (`x == true`) is
      // `x` itself; a specific nonzero integer constant additionally pins
      // the compared value (CmpValue) -- the constant evaluator computes
      // it, so enumerators and constexpr expressions pin like literals --
      // and getEdgeLockset() resolves the edges of the comparison against
      // each capability's exact success codes.
      bool TCond = false;
      const Expr *ConstSide = nullptr, *VarSide = nullptr;
      if (getStaticBooleanValue(BOP->getRHS(), TCond, *ASTCtx)) {
        ConstSide = BOP->getRHS();
        VarSide = BOP->getLHS();
      } else if (getStaticBooleanValue(BOP->getLHS(), TCond, *ASTCtx)) {
        ConstSide = BOP->getLHS();
        VarSide = BOP->getRHS();
      } else {
        return;
      }
      // Conversions seen so far were applied to this comparison's own
      // boolean result, not to the value it compares -- the contextual
      // conversion in `if (__builtin_expect(r == 2, 1))`, say. Only a
      // conversion found below, between the comparison and the call, can
      // cost the exact resolution.
      D.ValueNarrowed = false;
      // The comparison's operand value is materialized before the branch,
      // so a logical operator inside it was evaluated across a join.
      D.CrossedBlocks = true;
      // The compared value is the constant as the comparison sees it,
      // after its own promotions: `x == true` on an integer x is `x == 1`,
      // not a truthiness test, so the constant's own type does not decide
      // this. A boolean *result* is what makes `== 1` truthiness again,
      // and decodeTrylockBranch() applies that where the call is known.
      if (TCond) {
        // A truthy constant that is not an integer -- a floating-point or
        // pointer value the boolean evaluator accepted -- pins nothing
        // and is no truthiness test either: `r == 2.0` is true for one
        // result and false for every other, so neither edge may be read
        // as the plain branch's.
        Expr::EvalResult ER;
        if (ConstSide->isValueDependent() ||
            !ConstSide->EvaluateAsInt(ER, *ASTCtx))
          return;
        assert(!D.CmpValue && "a value comparison above was dropped on entry");
        D.CmpValue = ER.Val.getInt();
        D.CmpType = VarSide->IgnoreParenImpCasts()->getType();
        return decodeTrylockCond(VarSide, C, D);
      }
      D.Negate = !D.Negate;
      return decodeTrylockCond(VarSide, C, D);
    }
    if (BOP->getOpcode() == BO_LAnd || BOP->getOpcode() == BO_LOr) {
      // The right-hand side is evaluated in this very block, which is
      // reached only on the non-short-circuiting edge (the LHS was tested
      // in an earlier block), so the branch here is exactly a branch on
      // the RHS. That premise holds only for the block's own terminator
      // condition in situ: a walk that arrived through a stored variable,
      // a comparison, or a `?:` value merge sits at a join the
      // short-circuit edge also reaches, where the LHS may have produced
      // the value with the RHS never evaluated -- refuse the descent
      // there.
      if (D.CrossedBlocks)
        return;
      // What a comparison above compares is this operator's own boolean,
      // not the call's result: `(c && r) == 1` asks whether `c && r` is
      // true, and in C that boolean is an int, spelled exactly like a
      // success code. (`foldCompareToBoolean()` drops the comparison for the
      // one value the operator can produce and refuses every other.)
      if (!D.foldCompareToBoolean())
        return;
      // The short-circuit path may have left the LHS's block without
      // evaluating the RHS at all (see TrylockDecode::ShortCircuit).
      D.ShortCircuit = true;
      return decodeTrylockCond(BOP->getRHS(), C, D);
    }
    // An assignment used as a condition (`if ((b = mu.TryLock()))`)
    // evaluates to its right-hand side; the assigned value is
    // materialized, so a logical operator inside it crossed a join.
    if (BOP->getOpcode() == BO_Assign) {
      D.CrossedBlocks = true;
      return decodeTrylockCond(BOP->getRHS(), C, D);
    }
    return;
  } else if (const auto *COP = dyn_cast<ConditionalOperator>(Cond)) {
    // The `?:` arms merge at a value join, so every descent below leaves
    // the terminator condition.
    bool TCond, FCond;
    if (getMergedArmTruthiness(COP->getTrueExpr(), TCond, D, *ASTCtx) &&
        getMergedArmTruthiness(COP->getFalseExpr(), FCond, D, *ASTCtx)) {
      // Both arms are constants: the value the condition selects is one of
      // them, so its magnitude is the arm's and proves nothing about the
      // result's -- an exact value a later edge carries (a case label)
      // must not be applied to the result (ValueMerged clears
      // ValueIsResult). What the branch asks of that value is what decides
      // the condition. A comparison against a specific value is satisfied
      // exactly when the condition selected the arm holding it, whatever
      // the two arms' truthiness (`(t ? 1 : 2) == 1` is `t` as much as
      // `(t ? 1 : 0) == 1` is), and a value neither arm holds is never
      // equal, so it determines nothing. A plain branch instead reads the
      // value's truthiness, which is the condition's only where the arms
      // differ in it.
      if (D.CmpValue) {
        std::optional<bool> TrueArmHasIt =
            constantMeetsCond(COP->getTrueExpr(), TCond, D, *ASTCtx);
        std::optional<bool> FalseArmHasIt =
            constantMeetsCond(COP->getFalseExpr(), FCond, D, *ASTCtx);
        if (!TrueArmHasIt || !FalseArmHasIt || *TrueArmHasIt == *FalseArmHasIt)
          return;
        if (!*TrueArmHasIt)
          D.Negate = !D.Negate;
        D.CmpValue.reset();
      } else if (TCond == FCond) {
        return; // Constants of equal truthiness determine nothing.
      } else if (!TCond) {
        D.Negate = !D.Negate;
      }
      // Both arms were read through the conversion above, so the branch is
      // now exactly a branch on the condition: what the conversion does to
      // the merged value is settled, and only a conversion below this one
      // still stands between the condition and the call.
      D.ValueNarrowed = false;
      D.TruthinessLost = false;
      D.NarrowedTo = QualType();
      D.ValueMerged = true;
      D.CrossedBlocks = true;
      return decodeTrylockCond(COP->getCond(), C, D);
    }
    // One arm is a constant of truthiness K, the other is not: like the
    // merged variable above, a branch on the value still identifies the
    // non-constant arm -- on an edge where the value's truthiness is !K it
    // can only be that arm's result.
    bool ArmCond;
    const Expr *NonConstArm = nullptr, *ConstArm = nullptr;
    std::optional<bool> K;
    if (getStaticBooleanValue(COP->getTrueExpr(), ArmCond, *ASTCtx)) {
      K = ArmCond;
      ConstArm = COP->getTrueExpr();
      NonConstArm = COP->getFalseExpr();
    } else if (getStaticBooleanValue(COP->getFalseExpr(), ArmCond, *ASTCtx)) {
      K = ArmCond;
      ConstArm = COP->getFalseExpr();
      NonConstArm = COP->getTrueExpr();
    }
    if (!K)
      return; // Two non-constant arms determine nothing.
    // The non-constant arm may hold the call itself
    // (`c ? mu.TryLock() : false`): an edge whose value-truthiness is !K
    // proves that arm ran and carries its result, while an edge matching K
    // may carry the constant with the call never executed -- AmbiguousCond's
    // premise. Only one such merge can be resolved per condition.
    //
    // The arm takes priority over the condition below. The `?:` terminator
    // has already honored the condition's own branch
    // (decodeTrylockBranch()), so its result is resolved and its arm join
    // reconstituted; the arm's result has been branched on nowhere else,
    // and this is the only branch that can resolve it. It is also what the
    // `if` spelling of the same merge does, whose value identifies the arm
    // (`if (X) v = Y(); else v = false; if (v)` resolves Y).
    if (!D.AmbiguousCond) {
      // Which edge the constant arm can account for depends on what the
      // condition asks: its truthiness for a plain branch, but under a
      // comparison the edge the constant itself satisfies -- the same
      // rule the merged variable above follows (constantMeetsCond()).
      std::optional<bool> ConstMeetsCond =
          constantMeetsCond(ConstArm, *K, D, *ASTCtx);
      if (!ConstMeetsCond)
        return;
      TrylockDecode ArmD = D;
      ArmD.AmbiguousCond = *ConstMeetsCond != D.Negate;
      if (ProbeOperand(NonConstArm, ArmD, /*MustResolve=*/false))
        return;
    }
    // Otherwise the call may sit in the condition
    // (`mu.TryLock() ? other() : 0`): an edge whose value-truthiness is !K
    // proves the non-constant arm ran, i.e. the condition selected it --
    // the condition's truthiness there is whether that arm is the true arm
    // -- while an edge matching K determines nothing: the value may be the
    // constant (the condition selected the other arm), or the arm's result
    // happening to share the constant's truthiness (with the condition --
    // the call's result -- unrestricted). Record those edges in
    // UnknownCond; a comparison of the merged value against a specific
    // constant (CmpValue) identifies neither arm, so it is refused.
    if (!D.UnknownCond && !D.CmpValue) {
      TrylockDecode CondD = D;
      CondD.UnknownCond = *K != D.Negate;
      // On the determined edges the branch truthiness is !K, while the
      // condition's truthiness is whether the non-constant arm is the
      // true arm. The two disagree for the "crossed" shapes
      // (`X ? false : nc`, `X ? nc : true`): fold the mismatch into
      // Negate.
      if ((NonConstArm == COP->getTrueExpr()) == *K)
        CondD.Negate = !CondD.Negate;
      ProbeOperand(COP->getCond(), CondD, /*MustResolve=*/true);
    }
    return;
  } else if (const auto *BCO = dyn_cast<BinaryConditionalOperator>(Cond)) {
    // GNU `r ?: x` keeps r itself when it is truthy. With x a falsy
    // constant the value's truthiness is exactly r's; a truthy constant
    // makes the value unconditionally truthy (the branch determines
    // nothing). With x non-constant, x may hold the call itself
    // (`flag ?: mu.TryLock()`) -- a falsy value proves both operands
    // falsy, and a truthy one may be r without the call executing, which
    // is AmbiguousCond's premise. It takes priority over r for the reason
    // the arm does above. Failing that, the call sits in r, where a truthy
    // value no longer determines its result (x may have produced it):
    // those edges are recorded in UnknownCond as for the one-constant-arm
    // `?:` above.
    bool FCond;
    if (getStaticBooleanValue(BCO->getFalseExpr(), FCond, *ASTCtx)) {
      if (FCond)
        return;
      D.CrossedBlocks = true;
      return decodeTrylockCond(BCO->getCommon(), C, D);
    }
    if (!D.AmbiguousCond) {
      TrylockDecode ArmD = D;
      ArmD.AmbiguousCond = !D.Negate;
      if (ProbeOperand(BCO->getFalseExpr(), ArmD, /*MustResolve=*/false))
        return;
    }
    if (!D.UnknownCond && !D.CmpValue) {
      TrylockDecode CondD = D;
      CondD.UnknownCond = !D.Negate;
      ProbeOperand(BCO->getCommon(), CondD, /*MustResolve=*/true);
    }
    return;
  } else if (const auto *OVE = dyn_cast<OpaqueValueExpr>(Cond)) {
    // A BinaryConditionalOperator's terminator condition is the opaque
    // value bound to its shared operand (getTerminatorCondition() returns
    // BCO->getCond()). The operand itself may have been evaluated across
    // earlier blocks -- this block is its value join when it
    // short-circuits -- so the descent is not in situ.
    if (const Expr *SE = OVE->getSourceExpr()) {
      D.CrossedBlocks = true;
      return decodeTrylockCond(SE, C, D);
    }
  } else if (const auto *SE = dyn_cast<StmtExpr>(Cond)) {
    if (const auto *CS = SE->getSubStmt(); CS && !CS->body_empty()) {
      if (const auto *E = dyn_cast<Expr>(CS->body_back())) {
        // The statement expression's value is materialized within it, so
        // a logical operator there was evaluated across a join.
        D.CrossedBlocks = true;
        return decodeTrylockCond(E, C, D);
      }
    }
  }
}

ThreadSafetyAnalyzer::TerminatorTrylockCall
ThreadSafetyAnalyzer::getTerminatorTrylockCall(const CFGBlock *Block) {
  const TrylockBranch &B = decodeTrylockBranch(Block);
  return {B.TrylockCall, B.MergedCall};
}

/// Find the try-acquire call whose result the condition starting at
/// \p Block branches on. Unlike getTerminatorTrylockCall(), this looks
/// through short-circuit evaluation: in a compound condition such as
/// `while (i < n && !ok)`, \p Block tests only `i < n` and the branch on the
/// try-acquire result sits in a successor block of the condition.
///
/// With \p CheckAllPaths, also reports whether every outgoing path of
/// \p Block reaches a branch on that same call's result: a short-circuit
/// edge escapes its condition without evaluating the rest, but may itself
/// lead to another branch on the result (`if (c && b) ...; else if (b)`),
/// which is verified by walking each escape edge the same way. A caller
/// weakening a definitely-held fact on the strength of the rebranch needs
/// this: on an escaping path that never rebranches, the weakened fact
/// leaks unresolved (intersectAndWarn()).
ThreadSafetyAnalyzer::ConditionTrylockCall
ThreadSafetyAnalyzer::getConditionTrylockCallExpr(const CFGBlock *Block,
                                                  bool CheckAllPaths) {
  // The walk follows the successor edges of logical-operator terminators,
  // which stay within one condition expression, and the fall-through edge
  // of transition blocks (single successor, no terminator) -- e.g. where a
  // branch join meets a loop back edge, one hop before the loop condition
  // that rebranches on the merged variable. A transition block need not be
  // empty: its statements cannot invalidate the decode, which uses the
  // condition block's own ExitContext, and a write to the branched-on
  // variable in it makes the resolution itself refuse (decodeTrylockCond).
  // A fall-through edge can reach an earlier block (the transition block's
  // successor is the back edge's target), so the visited set keeps the walk
  // finite; it is shared with the escape walks below (any walk that fails
  // ends the search, and the all-paths check below describes how a merge
  // into a visited block resolves).
  llvm::SmallPtrSet<const CFGBlock *, 8> Visited;
  SmallVector<const CFGBlock *, 4> Escapes;
  const CallExpr *WalkMerged = nullptr;
  // The blocks whose terminator decodes to the call: their outgoing edges
  // are what must actually resolve the demoted fact.
  SmallVector<const CFGBlock *, 2> Deciders;
  auto Walk = [&](const CFGBlock *Block) -> const CallExpr * {
    while (Block) {
      if (!Visited.insert(Block).second) {
        TerminatorTrylockCall T = getTerminatorTrylockCall(Block);
        WalkMerged = T.MergedCall;
        if (T.TrylockCall)
          Deciders.push_back(Block);
        return T.TrylockCall;
      }
      if (TerminatorTrylockCall T = getTerminatorTrylockCall(Block);
          T.TrylockCall) {
        WalkMerged = T.MergedCall;
        Deciders.push_back(Block);
        return T.TrylockCall;
      }
      if (const auto *BOP =
              dyn_cast_or_null<BinaryOperator>(Block->getTerminatorStmt());
          BOP && BOP->isLogicalOp()) {
        // Evaluation of the condition continues on the not-short-circuiting
        // edge: the true edge for &&, the false edge for ||. The other edge
        // escapes the condition; remember it for the all-paths check.
        auto SI = Block->succ_begin();
        auto EscapeSI = SI;
        auto &Advance = BOP->getOpcode() == BO_LOr ? SI : EscapeSI;
        if (Advance == Block->succ_end())
          return nullptr; // A logical operator always has both successors.
        ++Advance;
        if (EscapeSI != Block->succ_end())
          if (const CFGBlock *Escape = EscapeSI->getReachableBlock())
            Escapes.push_back(Escape);
        Block = SI == Block->succ_end() ? nullptr : SI->getReachableBlock();
        continue;
      }
      if (!Block->getTerminatorStmt() && Block->succ_size() == 1) {
        Block = Block->succ_begin()->getReachableBlock();
        continue;
      }
      return nullptr;
    }
    return nullptr;
  };

  ConditionTrylockCall Result;
  const CallExpr *Exp = Result.TrylockCall = Walk(Block);
  Result.MergedCall = Exp ? WalkMerged : nullptr;
  if (CheckAllPaths) {
    Result.ResolvesAllPaths = Exp != nullptr;
    // Each escape edge must itself lead to a branch on the same call (its
    // own escapes accumulate and are checked in turn). An escape landing
    // directly on an already-visited block has merged into a path already
    // verified to reach the call. A walk that reaches a visited block only
    // deeper in stops with that block's terminator decode (the
    // shared-visited early return above), so it succeeds only when that
    // block itself branches on the call, and otherwise fails the all-paths
    // check conservatively.
    while (Exp && !Escapes.empty()) {
      const CFGBlock *Escape = Escapes.pop_back_val();
      if (Visited.count(Escape))
        continue; // Merged into an already-verified path.
      if (Walk(Escape) != Exp) {
        Result.ResolvesAllPaths = false;
        break;
      }
    }
    // Finding the branch is not enough: it has to decide some capability
    // on some edge. A switch on the result whose labels pin nothing
    // (`switch (ok) { default: }`) leaves every edge Unknown, so
    // getEdgeLockset() resolves nothing and a silently demoted hold would
    // leak unreported. Naming the call is not that test: an edge carries
    // the call whenever it pins the branched-on value's truthiness, while
    // what it proves per capability is the resolution -- which a `?:`
    // merge's undetermined direction, or a code-keyed capability under a
    // truthiness edge, leaves Unknown on every direction of a branch that
    // still names the call. A branch that resolves some of its edges keeps
    // the promise where it can; what escapes the others is the ordinary
    // conservatism the beta diagnostics report.
    for (const CFGBlock *D : Deciders) {
      if (!Result.ResolvesAllPaths)
        break;
      bool AnyEdgeResolves = false;
      for (CFGBlock::const_succ_iterator SI = D->succ_begin(),
                                         SE = D->succ_end();
           SI != SE && !AnyEdgeResolves; ++SI)
        if (const CFGBlock *Succ = SI->getReachableBlock()) {
          TrylockEdge Edge = resolveTrylockEdge(D, Succ);
          AnyEdgeResolves =
              Edge.Infeasible ||
              llvm::any_of(Edge.Caps, [](const TrylockEdgeCap &EC) {
                return EC.Resolution != CapResolution::Unknown;
              });
        }
      if (!AnyEdgeResolves)
        Result.ResolvesAllPaths = false;
    }
  }
  return Result;
}

/// Decode a try-acquire attribute's success value. An expression that does
/// not constant-evaluate reads as false.
static bool getTrySuccessValue(ASTContext &Ctx, const Expr *BrE) {
  bool Result;
  return BrE && getStaticBooleanValue(BrE, Result, Ctx) && Result;
}

CapProfile ThreadSafetyAnalyzer::getCapProfile(const TryAcquireCaps &Caps,
                                               const CapabilityExpr &Probe) {
  CapProfile P;
  auto MatchesAny = [&](const CapExprSet &S) {
    return llvm::any_of(
        S, [&](const CapabilityExpr &CE) { return Probe.matches(CE); });
  };
  P.Truthy = MatchesAny(Caps.TruthyExclusive) || MatchesAny(Caps.TruthyShared);
  P.Falsy = MatchesAny(Caps.FalsyExclusive) || MatchesAny(Caps.FalsyShared);
  for (const auto &[CE, Code] : Caps.ExactCodes)
    if (Probe.matches(CE))
      P.Codes.push_back(Code);
  P.AnyNonzero = P.Codes.empty() || MatchesAny(Caps.TruthyAny);
  return P;
}

/// Decode what the terminator of \p Block branches on: if it is the result of
/// a call to a function annotated with try_acquire_capability (possibly
/// negated, compared, merged, or stored in a local variable), return that call
/// together with the resolution each branch direction proves for each
/// capability recorded for it. Attributes may carry different success values,
/// so each is recorded on its own. The decode is memoized in the block's
/// CFGBlockInfo, and the returned reference stays valid for the rest of the
/// analysis.
const TrylockBranch &
ThreadSafetyAnalyzer::decodeTrylockBranch(const CFGBlock *Block) {
  // No try-acquire call is recorded anywhere in this function (the record
  // is complete before the lockset walk, recordTryAcquireCalls()), so no
  // decode can find one: skip the walk, and the cache, entirely.
  static const TrylockBranch NoBranch;
  if (TryAcquireCapsMap.empty())
    return NoBranch;

  const unsigned BlockID = Block->getBlockID();
  std::optional<TrylockBranch> &Memo = BlockInfo[BlockID].TryBranch;
  if (Memo)
    return *Memo;
  auto CacheMiss = [&]() -> const TrylockBranch & { return Memo.emplace(); };

  const Stmt *Cond = Block->getTerminatorCondition();
  if (!Cond)
    return CacheMiss();

  TrylockDecode D;
  D.UseBlock = Block;
  decodeTrylockCond(Cond, BlockInfo[BlockID].ExitContext, D);
  if (!D.TrylockCall)
    return CacheMiss();

  // Translate call truthiness to branch truthiness.
  TrylockBranch Result;
  Result.TrylockCall = D.TrylockCall;
  Result.MergedCall = D.MergedCall;
  Result.ShortCircuit = D.ShortCircuit;
  // A comparison against 1 on a provably boolean value is just a
  // truthiness test (`b == 1` is `b`): resolve it as the plain branch,
  // which also restores the exact failure edge (`!= 1` on a boolean value
  // is `== 0`). Against any other value the comparison is never true, so
  // neither edge says anything about the result -- resolving it as a
  // plain branch would read the always-taken edge as a failure. What has
  // to be boolean is the operand the constant is compared against, which
  // is not always the call: `_Bool ok = f();` narrows an int result, and
  // `int x = b();` widens a boolean one. A value reached through any
  // other narrowing conversion (`short q = r;`) is a truthiness test of
  // the converted copy on one edge only: where the comparison holds, the
  // copy is nonzero and so is the result; where it does not, the result
  // may be any other code -- unlike a boolean copy, whose `!= 1` is
  // `== 0`. That edge resolves nothing: see TrylockDecode::TruthinessLost
  // and the narrowed comparison below.
  bool OneSidedTruthy = D.TruthinessLost;
  if (D.CmpValue) {
    const bool BooleanCompared =
        D.TrylockCall->isKnownToHaveBooleanValue() ||
        (!D.CmpType.isNull() && D.CmpType->isBooleanType());
    if (BooleanCompared && !D.CmpValue->isOne())
      return CacheMiss();
    if (BooleanCompared) {
      D.CmpValue.reset();
    } else if (D.ValueNarrowed) {
      // A value the copy's own type cannot represent is one no copy ever
      // equals, so the comparison is never true: the edge it holds on is
      // unreachable and the other one is taken whatever the result was.
      // Neither says anything -- reading the equal edge as truthy below
      // would manufacture a hold on the unreachable one and carry it to
      // the join (`short q = f(); if (q == 65536)`).
      if (!D.NarrowedTo.isNull() &&
          D.NarrowedTo->isIntegralOrEnumerationType() &&
          !representableIn(*ASTCtx, *D.CmpValue, D.NarrowedTo))
        return CacheMiss();
      // The compared value is a narrowed copy, so the comparison pins no
      // code -- every result whose low bits are the value satisfies it
      // (`short q = f(); q == 1` for a result of 65537). What survives is
      // one direction of its truthiness: a nonzero copy is only ever a
      // nonzero result, since a conversion of zero is zero, so the equal
      // edge resolves as a truthy branch (the compared value is nonzero,
      // CmpValue is only recorded for a truthy constant). The other edge
      // resolves nothing: the result there may be truthy or falsy.
      OneSidedTruthy = true;
      D.CmpValue.reset();
    }
  }
  const auto MapIt = TryAcquireCapsMap.find(D.TrylockCall);
  // A fully-reconciled call (every capability moved to the unconditional
  // groups) creates no try facts, so a branch on its result proves
  // nothing and records nothing here.
  if (MapIt == TryAcquireCapsMap.end() || !MapIt->second.hasConditionalCap())
    return CacheMiss();
  const TryAcquireCaps &Caps = MapIt->second;
  // A call that declares no discriminating success code reports success by
  // truthiness alone (declaresExactSuccessCode()), so its values say
  // nothing its polarities do not: a comparison against a nonzero constant
  // is the truthiness test `if (r)` -- which is how the analysis read
  // every try-acquire before codes existed, and what `if (rc == 1)` on a
  // call declaring 1 means -- and a case label pins nothing.
  const bool ValueKeyed = !Caps.ExactCodes.empty();
  if (!ValueKeyed)
    D.CmpValue.reset();
  if (D.AmbiguousCond)
    (*D.AmbiguousCond ? Result.AmbiguousTrue : Result.AmbiguousFalse) = true;
  // A narrowed value is not the result either, so a switch over it
  // resolves no case label against the codes -- `switch (ok)` on a
  // `bool ok = f()` selects `case 1:` for every nonzero result -- and
  // neither is the boolean of a `&&` or `||` the walk descended through
  // (`switch (c && f())`, whose labels name that boolean), nor a value a
  // `?:` merge carries with the arm's own magnitude (ValueMerged) or
  // leaves undetermined on one direction (UnknownCond).
  Result.ValueIsResult = ValueKeyed && !D.Negate && !D.CmpValue &&
                         !D.ValueNarrowed && !D.ShortCircuit &&
                         !D.UnknownCond && !D.ValueMerged;
  Result.ValueCompared = D.CmpValue.has_value();

  // Per capability, on the direction where the condition reports the
  // result (Direct) and its inverse:
  //  * A branch comparing against a specific code (`result == code`)
  //    proves success for a capability whose region contains the value
  //    and failure for every other capability of the call: the declared
  //    codes discriminate the outcomes. The inverse direction only
  //    excludes that one value: failure for a capability whose whole
  //    region is that value, and nothing more -- in particular NOT that
  //    the result is falsy (a result of another code takes that edge with
  //    its capability still acquired).
  //  * A plain branch resolves by the success values' polarity, which
  //    decides a capability only where the polarity covers the edge: a
  //    capability keyed to codes, or acquired under both polarities,
  //    resolves only where an exact value does.
  // A direction recorded as undetermined by the decode (a `?:` merge, see
  // TrylockDecode::UnknownCond) resolves nothing for any capability.
  {
    auto AddCaps = [&](const CapExprSet &CapSet, LockKind LK) {
      for (const CapabilityExpr &CE : CapSet) {
        CapProfile P = getCapProfile(Caps, CE);
        CapResolution Direct, Inverse;
        if (D.CmpValue) {
          auto IsCmpValue = [&](const llvm::APSInt &C) {
            return llvm::APSInt::isSameValue(C, *D.CmpValue);
          };
          Direct = P.containsValue(*D.CmpValue) ? CapResolution::Success
                                                : CapResolution::Failure;
          Inverse = P.regionExcludedBy(IsCmpValue) ? CapResolution::Failure
                                                   : CapResolution::Unknown;
        } else {
          // A truthiness branch says only that the result is nonzero,
          // which proves an acquisition only where every nonzero result
          // makes it: a capability keyed to codes is acquired by some of
          // them and not by others, and the edge does not say which
          // (`if (TryCodes())`, where 1 acquires mu1 and 2 acquires mu2,
          // proves neither). Its fact stays for a comparison or a case
          // label to resolve. The falsy edge is exact either way: zero is
          // no code, and a falsy attribute reports its acquisition there.
          Direct = !P.Truthy                  ? CapResolution::Failure
                   : P.Falsy || !P.AnyNonzero ? CapResolution::Unknown
                                              : CapResolution::Success;
          Inverse = P.Falsy ? CapResolution::Success : CapResolution::Failure;
          if (OneSidedTruthy)
            Inverse = CapResolution::Unknown;
        }
        (D.Negate ? Result.OnFalse : Result.OnTrue).push_back({CE, LK, Direct});
        (D.Negate ? Result.OnTrue : Result.OnFalse)
            .push_back({CE, LK, Inverse});
        if (D.UnknownCond)
          (*D.UnknownCond ? Result.OnTrue : Result.OnFalse).back().Resolution =
              CapResolution::Unknown;
        Result.Profiles.push_back(std::move(P));
      }
    };
    AddCaps(Caps.TruthyExclusive, LK_Exclusive);
    AddCaps(Caps.TruthyShared, LK_Shared);
    AddCaps(Caps.FalsyExclusive, LK_Exclusive);
    AddCaps(Caps.FalsyShared, LK_Shared);
  }
  assert((!Result.OnTrue.empty() || !Result.OnFalse.empty()) &&
         "a conditional capability records a resolution on both directions");
  return Memo.emplace(std::move(Result));
}

/// Decode a truthy success value's exact integer code: the specific result
/// value on which the attribute reports acquisition, computed by the
/// constant evaluator (so enumerators and constexpr expressions key the
/// same way as literals) and converted to \p ResultTy, the type the call
/// reports it in. A bool-typed value (`true`) instead promises
/// acquisition on any nonzero result, and a value the evaluator cannot
/// compute falls back the same way; both return nullopt.
static std::optional<llvm::APSInt>
getTrySuccessCode(ASTContext &Ctx, const Expr *BrE, QualType ResultTy) {
  if (!BrE || BrE->isValueDependent() ||
      BrE->IgnoreParenImpCasts()->getType()->isBooleanType())
    return std::nullopt;
  Expr::EvalResult ER;
  if (!BrE->EvaluateAsInt(ER, Ctx) || ER.Val.getInt() == 0)
    return std::nullopt;
  // The value is written in its own type, while the result a condition
  // compares against it is the callee's: `unsigned f() TRY_ACQUIRE(-1, mu)`
  // reports success as 0xffffffff, and the code has to be recorded that way
  // for `r == -1u` to match it. Sema accepts any integer or bool constant,
  // so the conversion is the caller's semantics, not a diagnosis.
  llvm::APSInt Code = ER.Val.getInt();
  if (!ResultTy.isNull() && ResultTy->isIntegralOrEnumerationType()) {
    Code = Code.extOrTrunc(Ctx.getIntWidth(ResultTy));
    Code.setIsSigned(ResultTy->isSignedIntegerOrEnumerationType());
    // The conversion lost the value (a code wider than the result type,
    // or one a boolean result cannot report): no result can equal it, so
    // it keys nothing. Fall back to the truthiness reading the attribute
    // had before codes existed rather than refuse every acquisition.
    if (Code == 0)
      return std::nullopt;
  }
  return Code;
}

/// What an edge out of a terminator implies about the branched-on value.
enum class EdgeValue {
  False,      ///< The value is zero on this edge.
  True,       ///< The value is nonzero on this edge.
  Unknown,    ///< The edge does not determine the value.
  Infeasible, ///< The edge cannot be taken (e.g. the implicit default of a
              ///< switch that lists every value of a boolean condition).
};

/// The value range [Lo, Hi] a case label covers (a single value unless it
/// is a GNU case range).
static std::pair<llvm::APSInt, llvm::APSInt> getCaseRange(ASTContext &Ctx,
                                                          const CaseStmt *CS) {
  llvm::APSInt Lo = CS->getLHS()->EvaluateKnownConstInt(Ctx);
  llvm::APSInt Hi =
      CS->getRHS() ? CS->getRHS()->EvaluateKnownConstInt(Ctx) : Lo;
  return std::make_pair(Lo, Hi);
}

const SwitchSummary &
ThreadSafetyAnalyzer::getSwitchSummary(ASTContext &Ctx, const SwitchStmt *SW) {
  auto [It, Inserted] = SwitchSummaries.try_emplace(SW);
  SwitchSummary &Sum = It->second;
  if (!Inserted)
    return Sum;
  for (const SwitchCase *SC = SW->getSwitchCaseList(); SC;
       SC = SC->getNextSwitchCase()) {
    const auto *CS = dyn_cast<CaseStmt>(SC);
    if (!CS)
      continue;
    auto [Lo, Hi] = getCaseRange(Ctx, CS);
    Sum.OwnCases.try_emplace(CS, Lo, Hi);
    Sum.ZeroListed |= Lo <= 0 && Hi >= 0;
    Sum.OneListed |= Lo <= 1 && Hi >= 1;
  }
  // Not just bool-typed conditions: an int-typed condition provably 0/1
  // (e.g. a comparison in C) derives the same way.
  Sum.IsBool = SW->getCond()->isKnownToHaveBooleanValue();
  return Sum;
}

/// Determine the truthiness of a switch condition along the edge to
/// \p CaseBlock. Also reports the exact value information the edge
/// carries: a single-value nonzero case label pins the condition to that
/// value (\p EqValue), and the default edge excludes every listed label
/// range (\p Excluded); getEdgeLockset() resolves these against each
/// capability's exact success codes.
/// \p WantValues is false where the branched-on value is not the call's
/// result, so no exact value can refine anything: only the truthiness is
/// computed, and the ranges are not copied per edge.
static EdgeValue getSwitchEdgeValue(
    const SwitchSummary &Sum, const CFGBlock *CaseBlock, bool WantValues,
    std::optional<llvm::APSInt> &EqValue,
    SmallVectorImpl<std::pair<llvm::APSInt, llvm::APSInt>> &Excluded) {
  // A case label pins the value -- but only a label belonging to this
  // switch: the implicit fall-out successor can itself be a labeled
  // statement, e.g. a case of an enclosing switch that the fall-out edge
  // falls through into, which says nothing about this switch's condition
  // beyond matching none of its cases (the derivation below).
  auto OwnCase = Sum.OwnCases.end();
  if (const auto *CS = dyn_cast_if_present<CaseStmt>(CaseBlock->getLabel()))
    OwnCase = Sum.OwnCases.find(CS);
  if (OwnCase != Sum.OwnCases.end()) {
    // The range was evaluated once for the summary; do not re-run the
    // constant evaluator per edge.
    auto [Lo, Hi] = OwnCase->second;
    // A label a boolean condition can never match is a dead edge.
    if (Sum.IsBool && (Lo > 1 || Hi < 0))
      return EdgeValue::Infeasible;
    if (Lo == 0 && Hi == 0)
      return EdgeValue::False;
    if (Lo <= 0 && Hi >= 0)
      return EdgeValue::Unknown; // A GNU case range spanning zero and nonzero.
    if (WantValues && Lo == Hi)
      EqValue = Lo; // A single nonzero label pins the value exactly.
    return EdgeValue::True;
  }

  // The default edge (explicit, or the implicit fall-out successor): the
  // value matches none of the case labels. If zero is listed the value must
  // be nonzero; for a boolean condition with one listed it must be zero --
  // and with both listed this edge cannot be taken at all.
  if (WantValues)
    for (const auto &OwnCase : Sum.OwnCases)
      Excluded.push_back(OwnCase.second);
  if (Sum.ZeroListed)
    return Sum.IsBool && Sum.OneListed ? EdgeValue::Infeasible
                                       : EdgeValue::True;
  if (Sum.IsBool && Sum.OneListed)
    return EdgeValue::False;
  return EdgeValue::Unknown;
}

/// Decode what the edge from \p PredBlock to \p CurrBlock proves about
/// conditional capabilities, selected by the truthiness the edge assigns
/// to the branched-on value -- or, for a switch edge that pins the exact
/// branched-on value, by that value (see decodeTrylockBranch()). An edge
/// that does not determine the value reports no branch at all: the
/// try facts stay untouched either way.
ThreadSafetyAnalyzer::TrylockEdge
ThreadSafetyAnalyzer::resolveTrylockEdge(const CFGBlock *PredBlock,
                                         const CFGBlock *CurrBlock) {
  const TrylockBranch &B = decodeTrylockBranch(PredBlock);
  TrylockEdge Edge;
  if (!B.TrylockCall)
    return Edge;

  // Determine the truthiness of the branched-on value along this edge. An
  // if/loop terminator has a true and a false successor; each case label
  // of a switch pins the value. An edge that does not determine the value
  // reports no branch at all: the facts stay untouched either way.
  //
  // Alongside truthiness, collect the exact value information the edge
  // carries about the branched-on value: the value it pins it to
  // (EqValue), or the ranges it excludes it from (Excluded).
  EdgeValue CondVal = EdgeValue::Unknown;
  std::optional<llvm::APSInt> EqValue;
  SmallVector<std::pair<llvm::APSInt, llvm::APSInt>, 4> Excluded;
  if (const auto *SW =
          dyn_cast_if_present<SwitchStmt>(PredBlock->getTerminatorStmt())) {
    ASTContext &Ctx = B.TrylockCall->getCalleeDecl()->getASTContext();
    // The labels name the switched-on value, whose exact values apply to
    // the result only when the branched-on value is the result itself: a
    // negation or folded comparison in between (`switch (!ok)`,
    // `switch (r == 2)`) makes them say nothing exact about the result --
    // its truthiness still resolves through the capabilities' decoded
    // per-direction resolutions.
    CondVal = getSwitchEdgeValue(getSwitchSummary(Ctx, SW), CurrBlock,
                                 B.ValueIsResult, EqValue, Excluded);
  } else {
    bool TrueEdge = false, FalseEdge = false;
    int i = 0;
    for (CFGBlock::const_succ_iterator SI = PredBlock->succ_begin(),
                                       SE = PredBlock->succ_end();
         SI != SE && i < 2; ++SI, ++i)
      if (*SI == CurrBlock)
        (i == 0 ? TrueEdge : FalseEdge) = true;
    if (TrueEdge != FalseEdge)
      CondVal = TrueEdge ? EdgeValue::True : EdgeValue::False;
  }
  if (CondVal == EdgeValue::Infeasible) {
    // TrylockCall stays null: an edge that cannot be taken resolves
    // nothing, and Infeasible is meaningful on its own (getEdgeLockset()
    // tests it before the no-branch bail).
    Edge.Infeasible = true;
    return Edge;
  }
  if (CondVal == EdgeValue::Unknown && Excluded.empty())
    return Edge;

  Edge.TrylockCall = B.TrylockCall;
  Edge.MergedCall = B.MergedCall;
  Edge.ShortCircuit = B.ShortCircuit;
  // If the branched-on variable merges the call's result with a constant,
  // an edge matching the constant's truthiness does not prove the call
  // executed. A value edge (a switch label) carries no truthiness to tell
  // the sides apart, so any ambiguity in the branch reaches it.
  Edge.Ambiguous = CondVal == EdgeValue::True ? B.AmbiguousTrue
                   : CondVal == EdgeValue::False
                       ? B.AmbiguousFalse
                       : B.AmbiguousTrue || B.AmbiguousFalse;

  // Resolve one capability by the exact value the edge carries; nullopt
  // when that information does not decide it, in which case the caller
  // falls back to the capability's truthiness resolution.
  auto ResolveByValue =
      [&](const CapProfile &P) -> std::optional<CapResolution> {
    if (EqValue)
      return P.containsValue(*EqValue) ? CapResolution::Success
                                       : CapResolution::Failure;
    auto ValueExcluded = [&](const llvm::APSInt &V) {
      return llvm::any_of(Excluded, [&](const auto &R) {
        return llvm::APSInt::compareValues(V, R.first) >= 0 &&
               llvm::APSInt::compareValues(V, R.second) <= 0;
      });
    };
    if (P.regionExcludedBy(ValueExcluded))
      return CapResolution::Failure;
    return std::nullopt; // Beyond the exclusions, the truthiness decides.
  };
  assert(B.OnTrue.size() == B.Profiles.size() &&
         "one success-value profile per capability of the call");
  for (auto [TC, FC, P] : llvm::zip_equal(B.OnTrue, B.OnFalse, B.Profiles)) {
    std::optional<CapResolution> R;
    // Value reasoning cannot see through a merge. Where the branched-on
    // value may be the merge's constant, an exact value neither proves
    // nor disproves any code: the constant can stand in for a different
    // result of a call that did execute (`int s = 5; if (c) s = r;` --
    // the default edge carries 5 while r may have been 2). Truthiness
    // does not have this hole, because there the ambiguous edge is only
    // ever the constant's own truthiness and the other edge still pins
    // the result; so the plain per-direction resolutions still apply.
    // Only a switch loses every label this way: a comparison decides
    // which of its two edges the constant can account for and leaves the
    // other one exact (constantMeetsCond()), while the labels of a switch
    // are decided here, an edge at a time, with nothing to compare the
    // constant against.
    if ((EqValue || !Excluded.empty()) && !Edge.Ambiguous)
      R = ResolveByValue(P);
    if (!R)
      R = CondVal == EdgeValue::True    ? TC.Resolution
          : CondVal == EdgeValue::False ? FC.Resolution
                                        : CapResolution::Unknown;
    // The same hole in the per-direction resolutions of an `x == code`
    // branch, which are decided at decode time: on an ambiguous edge the
    // value may be the constant, so "not this code" is not "the call
    // failed". A truthiness branch keeps its resolutions, whichever way
    // they fall: there the value's truthiness is the same whether it is
    // the constant or the result, and each capability's own polarity has
    // already said what that proves for it.
    if (Edge.Ambiguous && B.ValueCompared && *R == CapResolution::Failure)
      R = CapResolution::Unknown;
    Edge.Caps.push_back({TC.Cap, TC.Kind, *R});
  }
  return Edge;
}

/// Find the lockset that holds on the edge between PredBlock
/// and CurrBlock.  The edge set is the exit set of PredBlock (passed
/// as the ExitSet parameter) plus any trylocks, which are conditionally held.
///
/// Returns true if the edge is infeasible: a resolved try fact of the
/// branched-on try-acquire says what its result is on every path into
/// PredBlock, so the edge implying the opposite cannot be taken. The caller
/// skips such edges at joins, like unreachable predecessors.
bool ThreadSafetyAnalyzer::getEdgeLockset(FactSet &Result,
                                          const FactSet &ExitSet,
                                          const CFGBlock *PredBlock,
                                          const CFGBlock *CurrBlock) {
  Result = ExitSet;

  TrylockEdge Edge = resolveTrylockEdge(PredBlock, CurrBlock);
  const CallExpr *Exp = Edge.TrylockCall;
  // Infeasibility stands on its own, before the call check: the flag must
  // not depend on TrylockCall also being set.
  if (Edge.Infeasible)
    return true;
  if (!Exp)
    return false;

  // If the branched-on variable merges the call's result with a constant,
  // an edge matching the constant's truthiness does not prove the call
  // executed. Each try fact decides for itself what such an edge still
  // proves: it resolves as a failure edge for one whose own attribute
  // reports no success here (even the call executing would mean failure
  // for that capability, and the call not executing means it was never
  // acquired), while one whose attribute reports success is left
  // untouched, like an unresolved condition -- attributes carry their own
  // success values, so one call's capabilities can split both ways across
  // the same edge. A ProvedNotHeld try fact likewise concludes no infeasibility
  // on such an edge: the edge may be taken with the constant's value, the call
  // never executed. Neither does a ProvedHeld one: it proves the call
  // executed and succeeded on every path into PredBlock, but not which of
  // the merge's two values the branched-on variable holds here, so a
  // constant-falsy edge stays feasible beside the proved hold.
  const bool Ambiguous = Edge.Ambiguous;

  // A try fact is re-identified by matching against the capabilities
  // recorded at the call, with the resolution this edge proves for each
  // (resolveTrylockEdge()): every try fact of the call was created from that
  // record.
  auto ResolveFact = [&](const CapabilityExpr &FE) {
    const auto *EC = llvm::find_if(
        Edge.Caps, [&](const TrylockEdgeCap &C) { return FE.matches(C.Cap); });
    if (EC != Edge.Caps.end())
      return EC->Resolution;
    // A hold of the capability a release-style try-acquire gives up
    // (try_acquire_capability(true, !mu) named !mu, this try fact is mu's):
    // the call's outcome resolves it inverted -- acquiring !mu releases mu,
    // failing to acquire it leaves the hold standing. A join demotes such
    // a hold to the call's try fact (intersectAndWarn()).
    CapabilityExpr Inverse = !FE;
    const auto *InvEC = llvm::find_if(Edge.Caps, [&](const TrylockEdgeCap &C) {
      return Inverse.matches(C.Cap);
    });
    assert(InvEC != Edge.Caps.end() &&
           "try-acquire fact does not match any capability of its call");
    if (InvEC == Edge.Caps.end())
      return CapResolution::Unknown;
    switch (InvEC->Resolution) {
    case CapResolution::Success:
      return CapResolution::Failure;
    case CapResolution::Failure:
      return CapResolution::Success;
    case CapResolution::Unknown:
      return CapResolution::Unknown;
    }
    llvm_unreachable("unhandled capability resolution");
  };

  // This edge resolves every try fact of this call, each with its own
  // attribute's polarity. A conditional try fact is folded into the
  // capability's definite fact on the branch on which its attribute reports
  // success -- one level deeper, or newly created -- and kept as the proof
  // of that level (ProvedHeld); on the other branch it records the failure
  // (ProvedNotHeld). It is resolved with the capability recorded at the call,
  // never a re-translation at this edge, which could name a different
  // capability (e.g. through a pointer reassigned since the call).
  //
  // A try fact already resolved by an earlier branch on the same result
  // says what the call's result is on every path into PredBlock: an edge
  // implying the opposite result cannot be taken, so the caller skips it
  // at joins like an unreachable predecessor (but still analyzes a block
  // this leaves without feasible predecessors, see runAnalysis()); on
  // other edges the resolved try fact and its definite fact are kept
  // unchanged -- re-resolving is not a new acquisition, so the hold keeps
  // its reentrancy depth and source, and the acquisition checks do not run
  // again. A Released try fact says less: its result is truthy on only some
  // paths, so it rules no edge out.
  SmallVector<const TryFactEntry *> Resolvable;
  // The branch tests "the result of that call" for the try facts of either
  // of two merged identical calls (Edge.MergedCall): a path that executed
  // the second holds only its try fact, and this branch resolves it.
  bool Infeasible = false;
  for (const auto &Fact : Result) {
    const auto *W = dyn_cast<TryFactEntry>(&FactMan[Fact]);
    if (!W || (W->origin() != Exp &&
               !(Edge.MergedCall && W->origin() == Edge.MergedCall)))
      continue;
    switch (W->state()) {
    case TryFactEntry::State::Conditional:
    case TryFactEntry::State::Released:
      Resolvable.push_back(W);
      break;
    case TryFactEntry::State::ProvedHeld:
      // The hold proved by this call's success: an edge on which the
      // capability's own attribute reports failure cannot be taken -- unless
      // the edge is ambiguous, where the branched-on value merges the call's
      // result with a constant. The call having succeeded says nothing about
      // which of the two the merge holds, so an edge the merge's constant
      // makes falsy is feasible beside the proved hold.
      if (!Ambiguous && ResolveFact(*W) == CapResolution::Failure)
        Infeasible = true;
      break;
    case TryFactEntry::State::ProvedNotHeld:
      // The call provably failed to acquire this try fact's capability on
      // every path here, so an edge on which the capability's own
      // attribute reports success cannot be taken; any other edge is
      // simply consistent with it (attributes carry their own success
      // values, so the test is per try fact, not per edge). An ambiguous
      // edge cannot be ruled out at all, since it does not prove the call
      // executed.
      if (!Ambiguous && ResolveFact(*W) == CapResolution::Success)
        Infeasible = true;
      break;
    }
  }
  // An infeasible edge's lockset still seeds the analysis of a block left
  // without feasible predecessors (runAnalysis()). It is deliberately left
  // UNRESOLVED, contradictions and all: the infeasibility proof rests on
  // the local-variable map, which can be stale (a captured mutation), and
  // then the "dead" block is live with exactly this state
  // (tryheld_recheck_after_captured_mutation). Genuinely dead arms pay
  // with residual over-reports against the contradiction-preserving state.
  if (Infeasible)
    return true;
  for (const TryFactEntry *W : Resolvable) {
    const CapResolution R = ResolveFact(*W);
    if (R == CapResolution::Unknown)
      continue; // The edge does not decide this capability's outcome.
    const bool Succeeds = R == CapResolution::Success;
    // An ambiguous edge does not prove the call executed, so it cannot
    // promote the try fact; it stays as it is, like an unresolved
    // condition.
    if (Succeeds && Ambiguous)
      continue;
    // A released try fact is a stale truth: the success edge resurrects
    // nothing. The failure edge excludes the paths its result was truthy
    // on, so what remains is the failure, recorded as for a conditional
    // try fact. A conditional try fact marked MayBeReleased (see
    // TryFactEntry::MayBeReleased) resolves the same way: its possible hold
    // cannot be promoted without resurrecting the released one, so the success
    // edge marks it Released.
    if (Succeeds && (W->released() || W->mayBeReleased())) {
      if (W->mayBeReleased())
        Result.replaceFact(
            FactMan, *W, W->withState(FactMan, TryFactEntry::State::Released));
      continue;
    }
    if (Succeeds) {
      const CapabilityExpr NegC = !*W;
      const FactEntry *Neg = Result.findDefinite(FactMan, NegC);
      // The successful release of a negative capability discharges one
      // level of the positive hold: with levels remaining the capability
      // is still held, so the try fact of the release resolves away instead
      // of proving a negative fact.
      if (W->negative() && Neg && isa<LockableFactEntry>(Neg)) {
        if (const FactEntry *ShallowerPos =
                cast<LockableFactEntry>(Neg)->leaveReentrant(FactMan)) {
          Result.replaceFact(FactMan, *Neg, ShallowerPos);
          // The try fact stays as the proof of the level this release
          // discharged, as on every other success edge: a second branch on
          // the same result re-resolves it instead of joining it afresh.
          Result.replaceFact(
              FactMan, *W,
              W->withState(FactMan, TryFactEntry::State::ProvedHeld));
          continue;
        }
      }
      // The try fact stays as the proof of the level it adds: joins and
      // later branches on the call's result recognize it (see
      // intersectAndWarn()). The acquisition checks ran at the call
      // (checkAcquiredCapability()); the proved acquisition now consumes
      // the negative capability the call could only require.
      Result.replaceFact(
          FactMan, *W, W->withState(FactMan, TryFactEntry::State::ProvedHeld));
      const FactEntry *Def = Result.findDefinite(FactMan, *W);
      if (const auto *Scope = dyn_cast_or_null<ScopedLockableFactEntry>(Def)) {
        // A guard's own try-acquire member succeeded here, so the guard now
        // holds what it manages. The scope's fact is not a hold to deepen:
        // it is the guard, which exists on either edge.
        for (const UnderlyingCapability &UM : Scope->getManaged())
          if (UM.Kind == UCK_Acquired)
            addLock(Result,
                    FactMan.createFact<LockableFactEntry>(
                        UM.Cap, W->kind(), W->loc(), FactEntry::Managed));
      } else if (Def) {
        // A negative fact has no levels: the proved release supersedes an
        // older negative for the capability rather than deepening it.
        Result.replaceFact(FactMan, *Def,
                           W->negative()
                               ? W->asDefinite(FactMan)
                               : cast<LockableFactEntry>(Def)->deepen(FactMan));
      } else {
        Result.addLock(FactMan, W->asDefinite(FactMan));
      }
      if (Neg) {
        Result.removeFact(FactMan, *Neg);
        // The proved release discharged the hold's last level (a remaining
        // one resolved the try fact away, above): whatever try-acquire's
        // success proved it is stale now, as after any other release
        // (handleUnlock()), so that a later branch on that result does not
        // resurrect the hold.
        if (W->negative())
          Result.releaseProved(FactMan, NegC, Exp->getExprLoc());
      } else if (W->negative()) {
        // The proved release of a merely conditionally held capability
        // released whichever call may have acquired it: their stored
        // results are stale (releaseConditional()).
        Result.releaseConditional(FactMan, NegC, Exp->getExprLoc());
      }
    } else {
      // Failure edge: the try fact records the failure, so a later branch
      // on the same result stays consistent (an edge implying success is
      // infeasible, above), and when no hold of the capability remains --
      // no definite hold, no other call's conditional try fact, no
      // conditional try fact of the negative capability (a release the call
      // may have proved, resolved in its own right) -- this edge proves
      // the capability not held: the negative fact is installed, unless
      // one already proves as much. The failure of a try-acquire of a
      // negative capability itself proves nothing about the positive
      // capability, so nothing is installed for it.
      const TryFactEntry *NotHeldW =
          W->withState(FactMan, TryFactEntry::State::ProvedNotHeld);
      // A value `?:` split its arms to make this edge: mark the failure
      // so their join reconstitutes silently even without beta.
      if (const auto *COp = dyn_cast_if_present<AbstractConditionalOperator>(
              PredBlock->getTerminatorStmt());
          COp && !COp->getType()->isVoidType())
        NotHeldW = NotHeldW->withArmSplit(FactMan);
      Result.replaceFact(FactMan, *W, NotHeldW);
      if (!W->negative() && !Result.findDefiniteOrConditional(FactMan, *W) &&
          !Result.anyConditional(FactMan, !*W))
        installNegativeFact(Result, FactMan, !*W, Exp->getExprLoc(),
                            /*KeepExisting=*/true);
    }
  }

  // Re-materialize a hold of the call that the analysis lost track of --
  // dropped at a join that could not reconstitute it from the two outcomes
  // of the one call -- on the edge where its attribute reports success: the
  // branch proves the call acquired the capability. It is keyed on the
  // acquisition throughout, since only the acquisition it proves may come
  // back: for a capability whose try fact this call really installed
  // (TrackedCaps), in the kind it installed it in, and only while the
  // call's stored result still determines a level of its own -- an
  // unconditional acquire, assert or release that consumed the try fact
  // took that over, and the hold it left is a hold of its own, whose
  // release the analysis need not still have a fact for
  // (FactManager::spentTryAcquire()). Refused, too, when a surviving fact
  // of the capability refutes the hold (refutesHoldOf()) or a definite fact
  // of the inverse capability does: a negative proves the hold released on
  // every path here, and for a try-release the positive hold means the
  // release already discharged a level. An ambiguous edge proves no
  // acquisition either way -- the call may never have executed -- so it
  // re-materializes nothing, and neither does one whose terminator reached
  // the call through a short-circuit operand: the path that short-circuited
  // never ran the call, and the edge does not tell the two apart
  // (`while (!(i >= n || ok))`, whose `||` is materialized for the `!`, so
  // both paths leave through the same terminator).
  if (auto MapIt = TryAcquireCapsMap.find(Exp);
      !Ambiguous && !Edge.ShortCircuit && MapIt != TryAcquireCapsMap.end() &&
      !FactMan.spentTryAcquire(Exp)) {
    // Success is decided per capability (a value edge can prove one
    // code's acquisition and another code's failure at once); the
    // per-capability resolution picks the ones this edge proves acquired.
    for (const TrylockEdgeCap &EC : Edge.Caps) {
      if (EC.Resolution != CapResolution::Success)
        continue;
      const CapabilityExpr &CE = EC.Cap;
      if (!MapIt->second.tracks(CE, EC.Kind))
        continue;
      if (Result.refutesHoldOf(FactMan, CE, Exp))
        continue;
      if (Result.findDefinite(FactMan, !CE))
        continue;
      auto *W = FactMan.createFact<TryFactEntry>(CE, EC.Kind, Exp->getExprLoc(),
                                                 FactEntry::Acquired, Exp);
      Result.addLock(FactMan,
                     W->withState(FactMan, TryFactEntry::State::ProvedHeld));
      Result.addLock(FactMan, W->asDefinite(FactMan));
    }
  }
  return false;
}

namespace {

/// We use this class to visit different types of expressions in
/// CFGBlocks, and build up the lockset.
/// An expression may cause us to add or remove locks from the lockset, or else
/// output error messages related to missing locks.
/// FIXME: In future, we may be able to not inherit from a visitor.
class BuildLockset : public ConstStmtVisitor<BuildLockset> {
  friend class ThreadSafetyAnalyzer;

  ThreadSafetyAnalyzer *Analyzer;
  FactSet FSet;
  // The fact set for the function on exit.
  const FactSet &FunctionExitFSet;

  /// A `LocalVariableMap::Context` wrapper that groups a context 'Q' with its
  /// immediate predecessor 'P' for a program point.  If the program point is
  /// right after a Stmt 'S', 'P' is the pre-context of 'S' and 'Q' is the
  /// post-context of 'S'.  Otherwise, 'P' == 'Q'.
  ///
  /// A DualLocalVarContext sets the global context for VarDefinition lookup to
  /// the post-context 'Q',  once CREATED or UPDATED to the next program
  /// point.  One can temporarily switch the global context to either 'P' or 'Q'
  /// using `switchToContextForScope`. The lifetime of the global context
  /// switching is bound to the enclosing scope. The global context will be set
  /// back to the prior state by the end of the scope.  This is done by the
  /// returned ContextSwitchScope object.
  ///
  /// Note: The pre- and post-context of a Stmt are distinct only in Beta mode
  /// (i.e., `Analyzer.Handler.issueBetaWarnings()`) because of the
  /// out-parameter validation.  If not in Beta mode, the global context for
  /// VarDefinition lookup is invisible, thus this wrapper has no impact on the
  /// analysis.
  class DualLocalVarContext {
  public:
    enum Point : char { Pre = 0, Post = 1 };

    class ContextSwitchScope {
      DualLocalVarContext &DC;
      Point LastPoint;

    public:
      ContextSwitchScope(DualLocalVarContext &DC, Point LastPoint)
          : DC(DC), LastPoint(LastPoint) {}
      ContextSwitchScope(const ContextSwitchScope &) = delete;
      ContextSwitchScope &operator=(const ContextSwitchScope &) = delete;
      ~ContextSwitchScope() { DC.switchContextTo(LastPoint); }
    };

    /// Temporarily switch context to \p P as long as the returned object lives.
    [[nodiscard]] ContextSwitchScope switchToContextForScope(Point P) {
      Point PriorPoint = CurrPoint;
      switchContextTo(P);
      return ContextSwitchScope(*this, PriorPoint);
    }

    /// Update the pre- and post-contexts to be associated with the next Stmt \p
    /// S. Set the global context to the post-context of \p S upon returning.
    ///
    /// If \p S is null, the behavior is as if the Stmt is a no-op--the
    /// post-context will shift to be the pre-context and the new post-context
    /// is the same as the old one, resulting in identical pre- and
    /// post-contexts.
    void moveToNextContext(const Stmt *S) {
      PrePost[Pre] = PrePost[Post];

      const LocalVariableMap::Context &NewPostCtx =
          S ? Analyzer.LocalVarMap.getNextContext(CtxIndex, S, *PrePost[Post])
            : *PrePost[Pre];

      PrePost[Post] = &NewPostCtx;
      switchContextTo(Post);
    }

    /// Constructs a DualLocalVarContext for the entry program point, where pre-
    /// and post-contexts are both equal to the \p EntryContext.
    DualLocalVarContext(ThreadSafetyAnalyzer &Analyzer, unsigned EntryIdx,
                        const LocalVariableMap::Context *EntryContext)
        : Analyzer(Analyzer), PrePost{EntryContext, EntryContext},
          CurrPoint(Post), CtxIndex(EntryIdx) {
      assert(EntryContext);
      switchContextTo(Post);
    }

  private:
    ThreadSafetyAnalyzer &Analyzer;
    // PrePost[0] points to the pre-context and
    // PrePost[1] points to the post-context:
    std::array<const LocalVariableMap::Context *, 2> PrePost;
    Point CurrPoint;
    unsigned CtxIndex;

    void switchContextTo(Point P) {
      if (!Analyzer.Handler.issueBetaWarnings())
        return;
      Analyzer.SxBuilder.setLookupLocalVarExpr(
          [Ctx = *PrePost[P],
           Analyzer = &Analyzer](const NamedDecl *D) mutable -> const Expr * {
            return Analyzer->LocalVarMap.lookupExpr(D, Ctx);
          });
      CurrPoint = P;
    }
  };

  DualLocalVarContext LVarCtx;

  // To update the context used in attr-expr translation.  If `S` is non-null,
  // the context is updated to the program point right after 'S'.
  void updateLocalVarMapCtx(const Stmt *S) { LVarCtx.moveToNextContext(S); }

  // helper functions

  void checkAccess(const Expr *Exp, AccessKind AK,
                   ProtectedOperationKind POK = POK_VarAccess) {
    Analyzer->checkAccess(FSet, Exp, AK, POK);
  }
  void checkPtAccess(const Expr *Exp, AccessKind AK,
                     ProtectedOperationKind POK = POK_VarAccess) {
    Analyzer->checkPtAccess(FSet, Exp, AK, POK);
  }

  void handleCall(const Expr *Exp, const NamedDecl *D,
                  til::SExpr *Self = nullptr,
                  SourceLocation Loc = SourceLocation());
  void examineArguments(const FunctionDecl *FD,
                        CallExpr::const_arg_iterator ArgBegin,
                        CallExpr::const_arg_iterator ArgEnd,
                        bool SkipFirstParam = false);

public:
  BuildLockset(ThreadSafetyAnalyzer *Anlzr, CFGBlockInfo &Info,
               const FactSet &FunctionExitFSet)
      : ConstStmtVisitor<BuildLockset>(), Analyzer(Anlzr), FSet(Info.EntrySet),
        FunctionExitFSet(FunctionExitFSet),
        LVarCtx(*Analyzer, Info.EntryIndex, &Info.EntryContext) {
    updateLocalVarMapCtx(nullptr);
  }

  ~BuildLockset() { Analyzer->SxBuilder.setLookupLocalVarExpr(nullptr); }
  BuildLockset(const BuildLockset &) = delete;
  BuildLockset &operator=(const BuildLockset &) = delete;

  void VisitUnaryOperator(const UnaryOperator *UO);
  void VisitBinaryOperator(const BinaryOperator *BO);
  void VisitCastExpr(const CastExpr *CE);
  void VisitCallExpr(const CallExpr *Exp);
  void VisitCXXConstructExpr(const CXXConstructExpr *Exp);
  void VisitDeclStmt(const DeclStmt *S);
  void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *Exp);
  void VisitReturnStmt(const ReturnStmt *S);
};

} // namespace

/// Warn if the LSet does not contain a lock sufficient to protect access
/// of at least the passed in AccessKind.
void ThreadSafetyAnalyzer::warnIfMutexNotHeld(
    const FactSet &FSet, const NamedDecl *D, const Expr *Exp, AccessKind AK,
    Expr *MutexExp, ProtectedOperationKind POK, til::SExpr *Self,
    SourceLocation Loc) {
  LockKind LK = getLockKindFromAccessKind(AK);
  CapabilityExpr Cp = SxBuilder.translateAttrExpr(MutexExp, D, Exp, Self);
  if (Cp.isInvalid()) {
    warnInvalidLock(Handler, MutexExp, D, Exp, Cp.getKind());
    return;
  } else if (Cp.shouldIgnore()) {
    return;
  }

  if (Cp.negative()) {
    // Negative capabilities act like locks excluded. A conditionally held
    // capability may be held, which violates the exclusion just the same.
    if (FSet.findDefinite(FactMan, !Cp) || FSet.anyConditional(FactMan, !Cp)) {
      Handler.handleFunExcludesLock(
          Cp.getKind(), D->getNameAsString(), (!Cp).toString(), Loc,
          /*MaybeHeld=*/!FSet.findDefinite(FactMan, !Cp));
      return;
    }

    // If this does not refer to a negative capability in the same class,
    // then stop here.
    if (!inCurrentScope(Cp))
      return;

    // Otherwise the negative requirement must be propagated to the caller.
    if (!FSet.findDefinite(FactMan, Cp))
      Handler.handleNegativeNotHeld(D, Cp.toString(), Loc);
    return;
  }

  const FactEntry *LDat = FSet.findDefiniteUniv(FactMan, Cp);
  bool NoError = true;
  if (!LDat) {
    // No exact match found.  Look for a partial match.
    LDat = FSet.findDefinitePartialMatch(FactMan, Cp);
    if (LDat) {
      // Warn that there's no precise match.
      std::string PartMatchStr = LDat->toString();
      StringRef   PartMatchName(PartMatchStr);
      Handler.handleMutexNotHeld(Cp.getKind(), D, POK, Cp.toString(), LK, Loc,
                                 &PartMatchName);
    } else {
      // Warn that there's no match at all.
      Handler.handleMutexNotHeld(Cp.getKind(), D, POK, Cp.toString(), LK, Loc);
    }
    NoError = false;
  }
  // Make sure the mutex we found is the right kind.
  if (NoError && LDat && !LDat->isAtLeast(LK)) {
    Handler.handleMutexNotHeld(Cp.getKind(), D, POK, Cp.toString(), LK, Loc);
  }
}

void ThreadSafetyAnalyzer::warnIfAnyMutexNotHeldForRead(
    const FactSet &FSet, const NamedDecl *D, const Expr *Exp,
    llvm::ArrayRef<Expr *> Args, ProtectedOperationKind POK,
    SourceLocation Loc) {
  SmallVector<CapabilityExpr, 2> Caps;
  for (auto *Arg : Args) {
    CapabilityExpr Cp = SxBuilder.translateAttrExpr(Arg, D, Exp, nullptr);
    if (Cp.isInvalid()) {
      warnInvalidLock(Handler, Arg, D, Exp, Cp.getKind());
      continue;
    }
    if (Cp.shouldIgnore())
      continue;
    const FactEntry *LDat = FSet.findDefiniteUniv(FactMan, Cp);
    if (LDat && LDat->isAtLeast(LK_Shared))
      return; // At least one held — read access is safe.
    // FIXME: try findDefinitePartialMatch as a fallback to support
    //        -Wno-thread-safety-precise, as warnIfMutexNotHeld does.
    Caps.push_back(Cp);
  }
  if (Caps.empty())
    return;
  // Materialize names only now that we know we are going to warn.
  SmallVector<std::string, 2> NameStorage;
  SmallVector<StringRef, 2> Names;
  for (const auto &Cp : Caps) {
    NameStorage.push_back(Cp.toString());
    Names.push_back(NameStorage.back());
  }
  Handler.handleGuardedByAnyReadNotHeld(D, POK, Names, Loc);
}

/// Warn if the LSet contains the given lock.
void ThreadSafetyAnalyzer::warnIfMutexHeld(const FactSet &FSet,
                                           const NamedDecl *D, const Expr *Exp,
                                           Expr *MutexExp, til::SExpr *Self,
                                           SourceLocation Loc) {
  CapabilityExpr Cp = SxBuilder.translateAttrExpr(MutexExp, D, Exp, Self);
  if (Cp.isInvalid()) {
    warnInvalidLock(Handler, MutexExp, D, Exp, Cp.getKind());
    return;
  } else if (Cp.shouldIgnore()) {
    return;
  }

  // A conditionally held capability may be held, which violates the exclusion
  // just the same.
  if (FSet.findDefinite(FactMan, Cp))
    Handler.handleFunExcludesLock(Cp.getKind(), D->getNameAsString(),
                                  Cp.toString(), Loc, /*MaybeHeld=*/false);
  else if (FSet.anyConditional(FactMan, Cp))
    Handler.handleFunExcludesLock(Cp.getKind(), D->getNameAsString(),
                                  Cp.toString(), Loc, /*MaybeHeld=*/true);
}

/// Checks guarded_by and pt_guarded_by attributes.
/// Whenever we identify an access (read or write) to a DeclRefExpr that is
/// marked with guarded_by, we must ensure the appropriate mutexes are held.
/// Similarly, we check if the access is to an expression that dereferences
/// a pointer marked with pt_guarded_by.
void ThreadSafetyAnalyzer::checkAccess(const FactSet &FSet, const Expr *Exp,
                                       AccessKind AK,
                                       ProtectedOperationKind POK) {
  Exp = Exp->IgnoreImplicit()->IgnoreParenCasts();

  SourceLocation Loc = Exp->getExprLoc();

  // Local variables of reference type cannot be re-assigned;
  // map them to their initializer.
  while (const auto *DRE = dyn_cast<DeclRefExpr>(Exp)) {
    const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()->getCanonicalDecl());
    if (VD && VD->isLocalVarDecl() && VD->getType()->isReferenceType()) {
      if (const auto *E = VD->getInit()) {
        // Guard against self-initialization. e.g., int &i = i;
        if (E == Exp)
          break;
        Exp = E->IgnoreImplicit()->IgnoreParenCasts();
        continue;
      }
    }
    break;
  }

  if (const auto *UO = dyn_cast<UnaryOperator>(Exp)) {
    // For dereferences
    if (UO->getOpcode() == UO_Deref)
      checkPtAccess(FSet, UO->getSubExpr(), AK, POK);
    return;
  }

  if (const auto *BO = dyn_cast<BinaryOperator>(Exp)) {
    switch (BO->getOpcode()) {
    case BO_PtrMemD: // .*
      return checkAccess(FSet, BO->getLHS(), AK, POK);
    case BO_PtrMemI: // ->*
      return checkPtAccess(FSet, BO->getLHS(), AK, POK);
    default:
      return;
    }
  }

  if (const auto *AE = dyn_cast<ArraySubscriptExpr>(Exp)) {
    checkPtAccess(FSet, AE->getLHS(), AK, POK);
    return;
  }

  if (const auto *ME = dyn_cast<MemberExpr>(Exp)) {
    if (ME->isArrow())
      checkPtAccess(FSet, ME->getBase(), AK, POK);
    else
      checkAccess(FSet, ME->getBase(), AK, POK);
  }

  const ValueDecl *D = getValueDecl(Exp);
  if (!D || !D->hasAttrs())
    return;

  if (D->hasAttr<GuardedVarAttr>() && FSet.holdsNoCapability(FactMan)) {
    Handler.handleNoMutexHeld(D, POK, AK, Loc);
  }

  for (const auto *I : D->specific_attrs<GuardedByAttr>()) {
    if (AK == AK_Written || I->args_size() == 1) {
      // Write requires all capabilities; single-arg read uses the normal
      // per-lock warning path.
      for (auto *Arg : I->args())
        warnIfMutexNotHeld(FSet, D, Exp, AK, Arg, POK, nullptr, Loc);
    } else {
      // Multi-arg read: holding any one of the listed capabilities is
      // sufficient (a writer must hold all, so any one prevents writes).
      warnIfAnyMutexNotHeldForRead(FSet, D, Exp, I->args(), POK, Loc);
    }
  }
}

/// Checks pt_guarded_by and pt_guarded_var attributes.
/// POK is the same  operationKind that was passed to checkAccess.
void ThreadSafetyAnalyzer::checkPtAccess(const FactSet &FSet, const Expr *Exp,
                                         AccessKind AK,
                                         ProtectedOperationKind POK) {
  // Strip off paren- and cast-expressions, checking if we encounter any other
  // operator that should be delegated to checkAccess() instead.
  while (true) {
    if (const auto *PE = dyn_cast<ParenExpr>(Exp)) {
      Exp = PE->getSubExpr();
      continue;
    }
    if (const auto *CE = dyn_cast<CastExpr>(Exp)) {
      if (CE->getCastKind() == CK_ArrayToPointerDecay) {
        // If it's an actual array, and not a pointer, then it's elements
        // are protected by GUARDED_BY, not PT_GUARDED_BY;
        checkAccess(FSet, CE->getSubExpr(), AK, POK);
        return;
      }
      Exp = CE->getSubExpr();
      continue;
    }
    break;
  }

  if (const auto *UO = dyn_cast<UnaryOperator>(Exp)) {
    if (UO->getOpcode() == UO_AddrOf) {
      // Pointer access via pointer taken of variable, so the dereferenced
      // variable is not actually a pointer.
      checkAccess(FSet, UO->getSubExpr(), AK, POK);
      return;
    }
  }

  // Pass by reference/pointer warnings are under a different flag.
  ProtectedOperationKind PtPOK = POK_VarDereference;
  switch (POK) {
  case POK_PassByRef:
    PtPOK = POK_PtPassByRef;
    break;
  case POK_ReturnByRef:
    PtPOK = POK_PtReturnByRef;
    break;
  case POK_PassPointer:
    PtPOK = POK_PtPassPointer;
    break;
  case POK_ReturnPointer:
    PtPOK = POK_PtReturnPointer;
    break;
  default:
    break;
  }

  const ValueDecl *D = getValueDecl(Exp);
  if (!D || !D->hasAttrs())
    return;

  if (D->hasAttr<PtGuardedVarAttr>() && FSet.holdsNoCapability(FactMan))
    Handler.handleNoMutexHeld(D, PtPOK, AK, Exp->getExprLoc());

  for (auto const *I : D->specific_attrs<PtGuardedByAttr>()) {
    if (AK == AK_Written || I->args_size() == 1) {
      // Write requires all capabilities; single-arg read uses the normal
      // per-lock warning path.
      for (auto *Arg : I->args())
        warnIfMutexNotHeld(FSet, D, Exp, AK, Arg, PtPOK, nullptr,
                           Exp->getExprLoc());
    } else {
      // Multi-arg read: holding any one of the listed capabilities is
      // sufficient (a writer must hold all, so any one prevents writes).
      warnIfAnyMutexNotHeldForRead(FSet, D, Exp, I->args(), PtPOK,
                                   Exp->getExprLoc());
    }
  }
}

/// Process a function call, method call, constructor call,
/// or destructor call.  This involves looking at the attributes on the
/// corresponding function/method/constructor/destructor, issuing warnings,
/// and updating the locksets accordingly.
///
/// FIXME: For classes annotated with one of the guarded annotations, we need
/// to treat const method calls as reads and non-const method calls as writes,
/// and check that the appropriate locks are held. Non-const method calls with
/// the same signature as const method calls can be also treated as reads.
///
/// \param Exp   The call expression.
/// \param D     The callee declaration.
/// \param Self  If \p Exp = nullptr, the implicit this argument or the argument
///              of an implicitly called cleanup function.
/// \param Loc   If \p Exp = nullptr, the location.
void BuildLockset::handleCall(const Expr *Exp, const NamedDecl *D,
                              til::SExpr *Self, SourceLocation Loc) {
  // Move to the call Stmt so that both pre- and post-context are available.
  updateLocalVarMapCtx(Exp);

  // Most function attributes are associated with the pre-context. Exceptions
  // are AcquireCapability and AssertCapability, which ensure some locks are
  // held after the call, and thus are associated with the post-context. They
  // will require a temporary switch to the post-context during handling.
  //
  // Parameter attributes are restricted to scoped objects, and thus are NOT
  // context-sensitive.
  auto PreContextForThisScope =
      LVarCtx.switchToContextForScope(DualLocalVarContext::Pre);
  CapExprSet ExclusiveLocksToAdd, SharedLocksToAdd;
  CapExprSet ExclusiveLocksToRemove, SharedLocksToRemove, GenericLocksToRemove;
  CapExprSet ScopedReqsAndExcludes;
  // Try-acquire capabilities of a call without an expression (a destructor
  // or cleanup function): there is no result to branch on, but a reconciled
  // unconditional acquisition still applies. Materialized only for such a
  // call, since the record of every other lives in TryAcquireCapsMap.
  std::optional<ThreadSafetyAnalyzer::TryAcquireCaps> NoExprTryCaps;

  // Figure out if we're constructing an object of scoped lockable class
  CapabilityExpr Scp;
  if (Exp) {
    assert(!Self);
    const auto *TagT = Exp->getType()->getAs<TagType>();
    if (D->hasAttrs() && TagT && Exp->isPRValue()) {
      til::LiteralPtr *Placeholder =
          Analyzer->SxBuilder.createThisPlaceholder();
      [[maybe_unused]] auto inserted =
          Analyzer->ConstructedObjects.insert({Exp, Placeholder});
      assert(inserted.second && "Are we visiting the same expression again?");
      if (isa<CXXConstructExpr>(Exp))
        Self = Placeholder;
      if (TagT->getDecl()->getMostRecentDecl()->hasAttr<ScopedLockableAttr>())
        Scp = CapabilityExpr(Placeholder, Exp->getType(), /*Neg=*/false);
    }

    assert(Loc.isInvalid());
    Loc = Exp->getExprLoc();
  }

  // Try-acquired capabilities were recorded before the walk for a CallExpr,
  // so only a constructor or an expression-less call (a destructor or cleanup
  // function) records here, where its object placeholder is available. The
  // conditional locks are added to our lockset below, from the record.
  ThreadSafetyAnalyzer::TryAcquireCaps *TryCaps = nullptr;
  if (D->hasAttr<TryAcquireCapabilityAttr>()) {
    if (!Exp || (isa<CXXConstructExpr>(Exp) &&
                 !Analyzer->TryAcquireCapsMap.contains(Exp))) {
      auto PostContextForThisScope =
          LVarCtx.switchToContextForScope(DualLocalVarContext::Post);
      TryCaps = &Analyzer->recordTryAcquireCall(
          Exp, D, Self, Exp ? nullptr : &NoExprTryCaps.emplace());
    } else {
      TryCaps = Analyzer->recordedTryAcquireCaps(Exp);
    }
  }

  for(const Attr *At : D->attrs()) {
    switch (At->getKind()) {
      // When we encounter a lock function, we need to add the lock to our
      // lockset.
      case attr::AcquireCapability: {
        auto PostContextForThisScope =
            LVarCtx.switchToContextForScope(DualLocalVarContext::Post);
        const auto *A = cast<AcquireCapabilityAttr>(At);
        Analyzer->getMutexIDs(A->isShared() ? SharedLocksToAdd
                                            : ExclusiveLocksToAdd,
                              A, Exp, D, Self);
        break;
      }

      // Try-acquired capabilities are recorded above, before this loop.
      case attr::TryAcquireCapability:
        break;

      // An assert will add a lock to the lockset, but will not generate
      // a warning if it is already there, and will not generate a warning
      // if it is not removed.
      case attr::AssertCapability: {
        auto PostContextForThisScope =
            LVarCtx.switchToContextForScope(DualLocalVarContext::Post);
        const auto *A = cast<AssertCapabilityAttr>(At);
        CapExprSet AssertLocks;
        Analyzer->getMutexIDs(AssertLocks, A, Exp, D, Self);
        for (const auto &AssertLock : AssertLocks)
          Analyzer->addLock(
              FSet, Analyzer->FactMan.createFact<LockableFactEntry>(
                        AssertLock, A->isShared() ? LK_Shared : LK_Exclusive,
                        Loc, FactEntry::Asserted));
        break;
      }

      // When we encounter an unlock function, we need to remove unlocked
      // mutexes from the lockset, and flag a warning if they are not there.
      case attr::ReleaseCapability: {
        const auto *A = cast<ReleaseCapabilityAttr>(At);
        if (A->isGeneric())
          Analyzer->getMutexIDs(GenericLocksToRemove, A, Exp, D, Self);
        else if (A->isShared())
          Analyzer->getMutexIDs(SharedLocksToRemove, A, Exp, D, Self);
        else
          Analyzer->getMutexIDs(ExclusiveLocksToRemove, A, Exp, D, Self);
        break;
      }

      case attr::RequiresCapability: {
        const auto *A = cast<RequiresCapabilityAttr>(At);
        for (auto *Arg : A->args()) {
          Analyzer->warnIfMutexNotHeld(FSet, D, Exp,
                                       A->isShared() ? AK_Read : AK_Written,
                                       Arg, POK_FunctionCall, Self, Loc);
          // use for adopting a lock
          if (!Scp.shouldIgnore())
            Analyzer->getMutexIDs(ScopedReqsAndExcludes, A, Exp, D, Self);
        }
        break;
      }

      case attr::LocksExcluded: {
        const auto *A = cast<LocksExcludedAttr>(At);
        for (auto *Arg : A->args()) {
          Analyzer->warnIfMutexHeld(FSet, D, Exp, Arg, Self, Loc);
          // use for deferring a lock
          if (!Scp.shouldIgnore())
            Analyzer->getMutexIDs(ScopedReqsAndExcludes, A, Exp, D, Self);
        }
        break;
      }

      // Ignore attributes unrelated to thread-safety
      default:
        break;
    }
  }

  // Recording reconciled the polarity groups (recordTryAcquireCall); the
  // capabilities the reconciliation moved out of them are acquired regardless
  // of the call's result: diagnose and add them unconditionally. The
  // diagnostic is emitted here in the walk rather than at recording, where
  // the handler and the call's location are, and once per visited call.
  if (TryCaps) {
    auto AddRegardless = [&](const CapExprSet &Unconditional,
                             CapExprSet &LocksToAdd) {
      for (const auto &M : Unconditional) {
        Analyzer->Handler.handleTryLockRegardlessOfResult(M.getKind(),
                                                          M.toString(), Loc, D);
        LocksToAdd.push_back_nodup(M);
      }
    };
    AddRegardless(TryCaps->UnconditionalExclusive, ExclusiveLocksToAdd);
    AddRegardless(TryCaps->UnconditionalShared, SharedLocksToAdd);
  }

  std::optional<CallExpr::const_arg_range> Args;
  if (Exp) {
    if (const auto *CE = dyn_cast<CallExpr>(Exp))
      Args = CE->arguments();
    else if (const auto *CE = dyn_cast<CXXConstructExpr>(Exp))
      Args = CE->arguments();
    else
      llvm_unreachable("Unknown call kind");
  }
  const auto *CalledFunction = dyn_cast<FunctionDecl>(D);
  if (CalledFunction && Args.has_value()) {
    for (auto [Param, Arg] : zip(CalledFunction->parameters(), *Args)) {
      if (isCallbackParam(Param))
        continue;
      CapExprSet DeclaredLocks;
      for (const Attr *At : Param->attrs()) {
        switch (At->getKind()) {
        case attr::AcquireCapability: {
          const auto *A = cast<AcquireCapabilityAttr>(At);
          Analyzer->getMutexIDs(A->isShared() ? SharedLocksToAdd
                                              : ExclusiveLocksToAdd,
                                A, Exp, D, Self);
          Analyzer->getMutexIDs(DeclaredLocks, A, Exp, D, Self);
          break;
        }

        case attr::ReleaseCapability: {
          const auto *A = cast<ReleaseCapabilityAttr>(At);
          if (A->isGeneric())
            Analyzer->getMutexIDs(GenericLocksToRemove, A, Exp, D, Self);
          else if (A->isShared())
            Analyzer->getMutexIDs(SharedLocksToRemove, A, Exp, D, Self);
          else
            Analyzer->getMutexIDs(ExclusiveLocksToRemove, A, Exp, D, Self);
          Analyzer->getMutexIDs(DeclaredLocks, A, Exp, D, Self);
          break;
        }

        case attr::RequiresCapability: {
          const auto *A = cast<RequiresCapabilityAttr>(At);
          for (auto *Arg : A->args())
            Analyzer->warnIfMutexNotHeld(FSet, D, Exp,
                                         A->isShared() ? AK_Read : AK_Written,
                                         Arg, POK_FunctionCall, Self, Loc);
          Analyzer->getMutexIDs(DeclaredLocks, A, Exp, D, Self);
          break;
        }

        case attr::LocksExcluded: {
          const auto *A = cast<LocksExcludedAttr>(At);
          for (auto *Arg : A->args())
            Analyzer->warnIfMutexHeld(FSet, D, Exp, Arg, Self, Loc);
          Analyzer->getMutexIDs(DeclaredLocks, A, Exp, D, Self);
          break;
        }

        default:
          break;
        }
      }
      if (DeclaredLocks.empty())
        continue;
      CapabilityExpr Cp(Analyzer->SxBuilder.translate(Arg, nullptr),
                        StringRef("mutex"), /*Neg=*/false, /*Reentrant=*/false);
      if (const auto *CBTE = dyn_cast<CXXBindTemporaryExpr>(Arg->IgnoreCasts());
          Cp.isInvalid() && CBTE) {
        if (auto Object = Analyzer->ConstructedObjects.find(CBTE->getSubExpr());
            Object != Analyzer->ConstructedObjects.end())
          Cp = CapabilityExpr(Object->second, StringRef("mutex"), /*Neg=*/false,
                              /*Reentrant=*/false);
      }
      const FactEntry *Fact = FSet.findDefinite(Analyzer->FactMan, Cp);
      if (!Fact) {
        Analyzer->Handler.handleMutexNotHeld(Cp.getKind(), D, POK_FunctionCall,
                                             Cp.toString(), LK_Exclusive,
                                             Exp->getExprLoc());
        continue;
      }
      const auto *Scope = cast<ScopedLockableFactEntry>(Fact);
      for (const auto &[a, b] :
           zip_longest(DeclaredLocks, Scope->getUnderlyingMutexes())) {
        if (!a.has_value()) {
          Analyzer->Handler.handleExpectFewerUnderlyingMutexes(
              Exp->getExprLoc(), D->getLocation(), Scope->toString(),
              b.value().getKind(), b.value().toString());
        } else if (!b.has_value()) {
          Analyzer->Handler.handleExpectMoreUnderlyingMutexes(
              Exp->getExprLoc(), D->getLocation(), Scope->toString(),
              a.value().getKind(), a.value().toString());
        } else if (!a.value().equals(b.value())) {
          Analyzer->Handler.handleUnmatchedUnderlyingMutexes(
              Exp->getExprLoc(), D->getLocation(), Scope->toString(),
              a.value().getKind(), a.value().toString(), b.value().toString());
          break;
        }
      }
    }
  }
  // Remove locks first to allow lock upgrading/downgrading.
  // FIXME -- should only fully remove if the attribute refers to 'this'.
  bool Dtor = isa<CXXDestructorDecl>(D);
  for (const auto &M : ExclusiveLocksToRemove)
    Analyzer->removeLock(FSet, M, Loc, Dtor, LK_Exclusive);
  for (const auto &M : SharedLocksToRemove)
    Analyzer->removeLock(FSet, M, Loc, Dtor, LK_Shared);
  for (const auto &M : GenericLocksToRemove)
    Analyzer->removeLock(FSet, M, Loc, Dtor, LK_Generic);

  // Add locks.
  FactEntry::SourceKind Source =
      !Scp.shouldIgnore() ? FactEntry::Managed : FactEntry::Acquired;
  for (const auto &M : ExclusiveLocksToAdd)
    Analyzer->addLock(FSet, Analyzer->FactMan.createFact<LockableFactEntry>(
                                M, LK_Exclusive, Loc, Source));
  for (const auto &M : SharedLocksToAdd)
    Analyzer->addLock(FSet, Analyzer->FactMan.createFact<LockableFactEntry>(
                                M, LK_Shared, Loc, Source));

  // Add conditional locks. A scoped lockable's construction acquires its
  // underlying capabilities conditionally too, as managed try facts: a
  // constructor has no result to branch on, but the guard's destructor
  // pairs exactly with the conditional acquisition -- it releases each
  // capability only if the guard holds it -- so it disarms the try fact
  // silently (handleUncheckedConditionalUnlock()).
  CapExprSet TryLocksManaged;
  if (Exp && TryCaps) {
    // Recorded as the walk reaches the call, not in the pre-walk: a branch
    // decoded before the call is walked -- a loop-top `if (ok)` above
    // `ok = mu.TryLock()` -- must not re-materialize a hold for an
    // acquisition that has not happened on the first iteration
    // (tryheld_retry_with_continue). A fresh execution records afresh, and
    // starts over from whatever spent the previous result.
    TryCaps->TrackedCaps.clear();
    Analyzer->FactMan.clearSpentTryAcquire(Exp);
    // A capability recorded under both polarities (specific-code truthy
    // plus falsy, kept conditional by the reconciliation above) tracks
    // one try fact. Its kind is fixed when it is created, while the outcome
    // that decides which attribute applies is only known per edge, so a
    // cross-kind pairing takes the weaker kind: one of the two outcomes
    // promises no more than a shared hold, and an exclusive try fact would
    // grant more than that outcome allows.
    //
    // Two attributes naming the capability in both kinds under the *same*
    // outcome are a contradictory declaration -- one result cannot promise
    // an exclusive and a shared hold -- and nothing diagnoses that at the
    // declaration. Both acquisitions therefore run, weaker kind first: the
    // shared try fact is the one that survives, and the exclusive one is
    // refused with "may already be held" at the call.
    //
    // A scoped object manages the try facts its construction created, and
    // only those, each once whatever its kinds: a capability the construction
    // also acquires definitely is managed by that acquisition, and one whose
    // try fact addTryLock() declined was never acquired conditionally, so the
    // destructor has nothing of its to release.
    //
    // The set lookups below are linear, so the questions that cannot have
    // an answer are not asked: a call whose attributes name one kind only
    // cannot pair kinds at all, and one that acquires nothing definitely
    // has no definite list to consult. That keeps the common annotation
    // -- one kind, one polarity -- linear in the capabilities rather than
    // quadratic.
    const bool AnyDefinite =
        !ExclusiveLocksToAdd.empty() || !SharedLocksToAdd.empty();
    const bool AnyShared =
        !TryCaps->TruthyShared.empty() || !TryCaps->FalsyShared.empty();
    const bool AnyExclusive =
        !TryCaps->TruthyExclusive.empty() || !TryCaps->FalsyExclusive.empty();
    const bool MayPairKinds = AnyShared && AnyExclusive;
    SmallVector<std::pair<CapabilityExpr, LockKind>, 2> TryLocksAdded;
    auto AddTry = [&](const CapExprSet &CapSet, LockKind GroupKind) {
      for (const CapabilityExpr &M : CapSet) {
        if (AnyDefinite &&
            (ExclusiveLocksToAdd.contains(M) || SharedLocksToAdd.contains(M)))
          continue;
        LockKind LK = GroupKind;
        if (MayPairKinds) {
          const bool OneOutcomeBothKinds =
              (TryCaps->TruthyExclusive.contains(M) &&
               TryCaps->TruthyShared.contains(M)) ||
              (TryCaps->FalsyExclusive.contains(M) &&
               TryCaps->FalsyShared.contains(M));
          if (!OneOutcomeBothKinds && (TryCaps->TruthyShared.contains(M) ||
                                       TryCaps->FalsyShared.contains(M)))
            LK = LK_Shared;
        }
        if (llvm::any_of(TryLocksAdded, [&](const auto &Added) {
              return Added.second == LK && Added.first.equals(M);
            }))
          continue;
        TryLocksAdded.emplace_back(M, LK);
        if (Analyzer->addTryLock(FSet, M, LK, Loc, Exp, Source)) {
          TryCaps->TrackedCaps.emplace_back(M, LK);
          TryLocksManaged.push_back_nodup(M);
        }
      }
    };
    AddTry(TryCaps->TruthyShared, LK_Shared);
    AddTry(TryCaps->FalsyShared, LK_Shared);
    AddTry(TryCaps->TruthyExclusive, LK_Exclusive);
    AddTry(TryCaps->FalsyExclusive, LK_Exclusive);
  }

  if (!Scp.shouldIgnore()) {
    // Add the managing object as a dummy mutex, mapped to the underlying mutex.
    auto *ScopedEntry = Analyzer->FactMan.createFact<ScopedLockableFactEntry>(
        Scp, Loc, FactEntry::Acquired,
        ExclusiveLocksToAdd.size() + SharedLocksToAdd.size() +
            TryLocksManaged.size() + ScopedReqsAndExcludes.size() +
            ExclusiveLocksToRemove.size() + SharedLocksToRemove.size());
    for (const auto &M : ExclusiveLocksToAdd)
      ScopedEntry->addLock(M);
    for (const auto &M : SharedLocksToAdd)
      ScopedEntry->addLock(M);
    for (const auto &M : TryLocksManaged)
      ScopedEntry->addLock(M);
    // The destructor removes the try facts this construction created,
    // which it finds by their origin (unlock()).
    if (!TryLocksManaged.empty())
      ScopedEntry->setCondAcquireExpr(Exp);
    for (const auto &M : ScopedReqsAndExcludes)
      ScopedEntry->addLock(M);
    for (const auto &M : ExclusiveLocksToRemove)
      ScopedEntry->addExclusiveUnlock(M);
    for (const auto &M : SharedLocksToRemove)
      ScopedEntry->addSharedUnlock(M);
    Analyzer->addLock(FSet, ScopedEntry);
  }
}

/// For unary operations which read and write a variable, we need to
/// check whether we hold any required mutexes. Reads are checked in
/// VisitCastExpr.
void BuildLockset::VisitUnaryOperator(const UnaryOperator *UO) {
  switch (UO->getOpcode()) {
    case UO_PostDec:
    case UO_PostInc:
    case UO_PreDec:
    case UO_PreInc:
      checkAccess(UO->getSubExpr(), AK_Written);
      // The increment cleared the variable's definition
      // (VarMapBuilder::VisitUnaryOperator()); consume that context, or the
      // sequential cursor never reaches it and every later context saved in
      // this block is unreachable too -- capability translation would stay
      // frozen at the state before the increment.
      updateLocalVarMapCtx(UO);
      break;
    default:
      break;
  }
}

/// For binary operations which assign to a variable (writes), we need to check
/// whether we hold any required mutexes.
/// FIXME: Deal with non-primitive types.
void BuildLockset::VisitBinaryOperator(const BinaryOperator *BO) {
  if (!BO->isAssignmentOp())
    return;
  checkAccess(BO->getLHS(), AK_Written);
  updateLocalVarMapCtx(BO);
}

/// Whenever we do an LValue to Rvalue cast, we are reading a variable and
/// need to ensure we hold any required mutexes.
/// FIXME: Deal with non-primitive types.
void BuildLockset::VisitCastExpr(const CastExpr *CE) {
  if (CE->getCastKind() != CK_LValueToRValue)
    return;
  checkAccess(CE->getSubExpr(), AK_Read);
}

void BuildLockset::examineArguments(const FunctionDecl *FD,
                                    CallExpr::const_arg_iterator ArgBegin,
                                    CallExpr::const_arg_iterator ArgEnd,
                                    bool SkipFirstParam) {
  // Currently we can't do anything if we don't know the function declaration.
  if (!FD)
    return;

  // NO_THREAD_SAFETY_ANALYSIS does double duty here.  Normally it
  // only turns off checking within the body of a function, but we also
  // use it to turn off checking in arguments to the function.  This
  // could result in some false negatives, but the alternative is to
  // create yet another attribute.
  if (FD->hasAttr<NoThreadSafetyAnalysisAttr>())
    return;

  const ArrayRef<ParmVarDecl *> Params = FD->parameters();
  auto Param = Params.begin();
  if (SkipFirstParam)
    ++Param;

  // There can be default arguments, so we stop when one iterator is at end().
  for (auto Arg = ArgBegin; Param != Params.end() && Arg != ArgEnd;
       ++Param, ++Arg) {
    QualType Qt = (*Param)->getType();
    if (Qt->isReferenceType())
      checkAccess(*Arg, AK_Read, POK_PassByRef);
    else if (Qt->isPointerType())
      checkPtAccess(*Arg, AK_Read, POK_PassPointer);
  }
}

void BuildLockset::VisitCallExpr(const CallExpr *Exp) {
  if (const auto *CE = dyn_cast<CXXMemberCallExpr>(Exp)) {
    const auto *ME = dyn_cast<MemberExpr>(CE->getCallee());
    // ME can be null when calling a method pointer
    const CXXMethodDecl *MD = CE->getMethodDecl();

    if (ME && MD) {
      if (ME->isArrow()) {
        // Should perhaps be AK_Written if !MD->isConst().
        checkPtAccess(CE->getImplicitObjectArgument(), AK_Read);
      } else {
        // Should perhaps be AK_Written if !MD->isConst().
        checkAccess(CE->getImplicitObjectArgument(), AK_Read);
      }
    }

    examineArguments(CE->getDirectCallee(), CE->arg_begin(), CE->arg_end());
  } else if (const auto *OE = dyn_cast<CXXOperatorCallExpr>(Exp)) {
    OverloadedOperatorKind OEop = OE->getOperator();
    switch (OEop) {
      case OO_Equal:
      case OO_PlusEqual:
      case OO_MinusEqual:
      case OO_StarEqual:
      case OO_SlashEqual:
      case OO_PercentEqual:
      case OO_CaretEqual:
      case OO_AmpEqual:
      case OO_PipeEqual:
      case OO_LessLessEqual:
      case OO_GreaterGreaterEqual:
        checkAccess(OE->getArg(1), AK_Read);
        [[fallthrough]];
      case OO_PlusPlus:
      case OO_MinusMinus:
        checkAccess(OE->getArg(0), AK_Written);
        break;
      case OO_Star:
      case OO_ArrowStar:
      case OO_Arrow:
      case OO_Subscript:
        if (!(OEop == OO_Star && OE->getNumArgs() > 1)) {
          // Grrr.  operator* can be multiplication...
          checkPtAccess(OE->getArg(0), AK_Read);
        }
        [[fallthrough]];
      default: {
        // TODO: get rid of this, and rely on pass-by-ref instead.
        const Expr *Obj = OE->getArg(0);
        checkAccess(Obj, AK_Read);
        // Check the remaining arguments. For method operators, the first
        // argument is the implicit self argument, and doesn't appear in the
        // FunctionDecl, but for non-methods it does.
        const FunctionDecl *FD = OE->getDirectCallee();
        examineArguments(FD, std::next(OE->arg_begin()), OE->arg_end(),
                         /*SkipFirstParam*/ !isa<CXXMethodDecl>(FD));
        break;
      }
    }
  } else {
    examineArguments(Exp->getDirectCallee(), Exp->arg_begin(), Exp->arg_end());
  }

  auto *D = dyn_cast_or_null<NamedDecl>(Exp->getCalleeDecl());

  if (D)
    handleCall(Exp, D);
  else
    // Even if we cannot handle the call, we need to update the context for the
    // Stmt:
    updateLocalVarMapCtx(Exp);
}

void BuildLockset::VisitCXXConstructExpr(const CXXConstructExpr *Exp) {
  const CXXConstructorDecl *D = Exp->getConstructor();
  if (D && D->isCopyConstructor()) {
    const Expr* Source = Exp->getArg(0);
    checkAccess(Source, AK_Read);
  } else {
    examineArguments(D, Exp->arg_begin(), Exp->arg_end());
  }
  if (D && D->hasAttrs())
    handleCall(Exp, D);
}

static const Expr *UnpackConstruction(const Expr *E) {
  if (auto *CE = dyn_cast<CastExpr>(E))
    if (CE->getCastKind() == CK_NoOp)
      E = CE->getSubExpr()->IgnoreParens();
  if (auto *CE = dyn_cast<CastExpr>(E))
    if (CE->getCastKind() == CK_ConstructorConversion ||
        CE->getCastKind() == CK_UserDefinedConversion)
      E = CE->getSubExpr();
  if (auto *BTE = dyn_cast<CXXBindTemporaryExpr>(E))
    E = BTE->getSubExpr();
  return E;
}

void BuildLockset::VisitDeclStmt(const DeclStmt *S) {
  for (auto *D : S->getDeclGroup()) {
    if (auto *VD = dyn_cast_or_null<VarDecl>(D)) {
      const Expr *E = VD->getInit();
      if (!E)
        continue;
      E = E->IgnoreParens();

      // handle constructors that involve temporaries
      if (auto *EWC = dyn_cast<ExprWithCleanups>(E))
        E = EWC->getSubExpr()->IgnoreParens();
      E = UnpackConstruction(E);

      if (auto Object = Analyzer->ConstructedObjects.find(E);
          Object != Analyzer->ConstructedObjects.end()) {
        Object->second->setClangDecl(VD);
        Analyzer->ConstructedObjects.erase(Object);
      }
    }
  }
  updateLocalVarMapCtx(S);
}

void BuildLockset::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *Exp) {
  if (const ValueDecl *ExtD = Exp->getExtendingDecl()) {
    if (auto Object = Analyzer->ConstructedObjects.find(
            UnpackConstruction(Exp->getSubExpr()));
        Object != Analyzer->ConstructedObjects.end()) {
      Object->second->setClangDecl(ExtD);
      Analyzer->ConstructedObjects.erase(Object);
    }
  }
}

void BuildLockset::VisitReturnStmt(const ReturnStmt *S) {
  if (Analyzer->CurrentFunction == nullptr)
    return;
  const Expr *RetVal = S->getRetValue();
  if (!RetVal)
    return;

  // If returning by reference or pointer, check that the function requires the
  // appropriate capabilities.
  const QualType ReturnType =
      Analyzer->CurrentFunction->getReturnType().getCanonicalType();
  if (ReturnType->isLValueReferenceType()) {
    Analyzer->checkAccess(
        FunctionExitFSet, RetVal,
        ReturnType->getPointeeType().isConstQualified() ? AK_Read : AK_Written,
        POK_ReturnByRef);
  } else if (ReturnType->isPointerType()) {
    Analyzer->checkPtAccess(
        FunctionExitFSet, RetVal,
        ReturnType->getPointeeType().isConstQualified() ? AK_Read : AK_Written,
        POK_ReturnPointer);
  }
}

/// The reentrancy depth of the definite fact \p FE: how many times its
/// capability was re-acquired over the initial acquisition.
static unsigned reentrancyDepth(const FactEntry &FE) {
  const auto *LFE = dyn_cast<LockableFactEntry>(&FE);
  return LFE ? LFE->getReentrancyDepth() : 0;
}

/// Given two facts merging on a join point, possibly warn and decide whether to
/// keep or replace.
///
/// \return  false if we should keep \p A, true if we should take \p B.
bool ThreadSafetyAnalyzer::join(const FactEntry &A, const FactEntry &B,
                                SourceLocation JoinLoc,
                                LockErrorKind EntryLEK) {
  // Whether we can replace \p A by \p B.
  const bool CanModify = EntryLEK != LEK_LockedSomeLoopIterations;
  assert(!isa<TryFactEntry>(A) && !isa<TryFactEntry>(B) &&
         "try facts join through their own lattice");
  const unsigned ReentrancyDepthA = reentrancyDepth(A);
  const unsigned ReentrancyDepthB = reentrancyDepth(B);

  if (ReentrancyDepthA != ReentrancyDepthB) {
    Handler.handleMutexHeldEndOfScope(B.getKind(), B.toString(), B.loc(),
                                      JoinLoc, EntryLEK,
                                      /*ReentrancyMismatch=*/true);
    // Pick the FactEntry with the greater reentrancy depth as the "good"
    // fact to reduce potential later warnings.
    return CanModify && ReentrancyDepthA < ReentrancyDepthB;
  } else if (A.kind() != B.kind()) {
    // For managed capabilities, the destructor should unlock in the right mode
    // anyway. For asserted capabilities no unlocking is needed.
    if ((A.managed() || A.asserted()) && (B.managed() || B.asserted())) {
      // The shared capability subsumes the exclusive capability, if possible.
      bool ShouldTakeB = B.kind() == LK_Shared;
      if (CanModify || !ShouldTakeB)
        return ShouldTakeB;
    }
    Handler.handleExclusiveAndShared(B.getKind(), B.toString(), B.loc(),
                                     A.loc());
    // Take the exclusive capability to reduce further warnings.
    return CanModify && B.kind() == LK_Exclusive;
  } else {
    // The non-asserted capability is the one we want to track.
    return CanModify && A.asserted() && !B.asserted();
  }
}

namespace {

/// The join of one predecessor's exit set into a block's entry set: the
/// intersection of the two locksets, with warnings for what it loses.
///
/// Facts are paired by form: a capability's definite facts join through join(),
/// and its try facts join per identity -- two conditional try facts of one call
/// are identical, and try facts of different calls coexist, each still resolved
/// by the branch on its own result. The interplay is where one side holds the
/// capability definitely and the other only conditionally (a "mixed" join): the
/// merged state keeps the conditional side's facts, and the definite side's
/// extra level is diagnosed as lost, or -- when both sides also hold definite
/// levels -- as a reentrancy-depth mismatch.
///
/// A one-sided definite hold is kept, demoted to the try-acquire's conditional
/// try fact, under the rebranch exemption: the joining block's terminator
/// branches on the result of the call whose ProvedHeld try fact stands beside
/// the hold, so the paths re-diverge there and the outgoing edges re-resolve
/// the try fact. A ProvedHeld try fact on one side only is otherwise not
/// carried: the merged hold, if any, is not proved by that call.
///
/// A loop join compares against an entry set analyzed previously: it diagnoses,
/// but must not rewrite that set.
class LocksetJoin {
  ThreadSafetyAnalyzer &Analyzer;
  FactManager &FactMan;
  ThreadSafetyHandler &Handler;
  const JoinContext &Ctx;
  /// The entry set being computed, and a copy of it as it was before this join:
  /// the pairing reads the original while the merged set is rewritten in place.
  FactSet &EntrySet;
  const FactSet EntrySetOrig;
  const FactSet &ExitSet;

public:
  LocksetJoin(ThreadSafetyAnalyzer &Analyzer, FactSet &EntrySet,
              const FactSet &ExitSet, const JoinContext &Ctx)
      : Analyzer(Analyzer), FactMan(Analyzer.FactMan),
        Handler(Analyzer.Handler), Ctx(Ctx), EntrySet(EntrySet),
        EntrySetOrig(EntrySet), ExitSet(ExitSet) {}

  void run();

private:
  /// \name Exemptions
  /// \{
  /// Whether the joining block's terminator branches on the result of the
  /// call \p C: the rebranched call, or the second of two identical calls
  /// whose merged result it branches on (JoinContext::RebranchMergedCall)
  /// -- the outgoing edges resolve the try facts of either.
  bool isRebranchedCall(const Expr *C) const {
    return Ctx.RebranchTryLock &&
           (C == Ctx.RebranchTryLock || C == Ctx.RebranchMergedCall);
  }
  bool isTrylockRebranched(const TryFactEntry &W) const {
    return isRebranchedCall(W.origin());
  }
  const TryFactEntry *rebranchProof(const FactSet &Set,
                                    const FactEntry &Def) const;
  bool isRebranchedInverseHold(const FactSet &OtherSet,
                               const FactEntry &FE) const;
  bool rebranchVetoedByReleased(const FactSet &OtherSet,
                                const FactEntry &FE) const;
  /// Whether a branch on \p W's call can still occur after this join. Only
  /// a result the function stored in a local variable can be named by a
  /// later condition (LocalVariableMap::storesResultOf()); one used
  /// directly as a condition is resolved by that terminator and by no
  /// other, so its record has nothing left to prove.
  bool resultCanBeRebranched(const TryFactEntry &W) const {
    const auto *Call = dyn_cast<CallExpr>(W.origin());
    return !Call || Analyzer.LocalVarMap.storesResultOf(Call);
  }
  const TryFactEntry *sameOriginProof(const FactSet &OwnSet,
                                      const FactSet &OtherSet,
                                      const FactEntry &FE) const;
  bool sameOriginGate(LockErrorKind LEK, const TryFactEntry *NotHeldW) const;
  /// The ProvedNotHeld try fact of \p Cap from \p Call in \p Set, if any.
  const TryFactEntry *failureRecord(const FactSet &Set,
                                    const CapabilityExpr &Cap,
                                    const Expr *Call) const {
    return Set.findTryFactFrom(FactMan, Cap, Call,
                               TryFactEntry::State::ProvedNotHeld);
  }
  /// How a one-sided definite hold is demoted to conditional rather than
  /// lost, if it is: the terminator rebranches on the try-acquire that
  /// proved it (getEdgeLockset() will re-resolve it on the outgoing
  /// edges), unless that call's stale result on the other side vetoes it;
  /// or else the other side carries the call's own failure record (the
  /// same-origin form, which needs the other side's negative fact itself).
  /// Decided once per hold: the joins consult both fields below, and the
  /// predicates are not free.
  struct Demotion {
    /// The call whose conditional try fact the hold demotes to.
    const Expr *Origin = nullptr;
    /// Under the rebranch exemption (else the same-origin one).
    bool Rebranch = false;
    /// Whether what the call proved is the *release* of the capability, so
    /// that the demoted level is held while the call's result is falsy.
    /// The conditional try fact carrying it is then the call's own, keyed
    /// on the negative capability.
    bool Inverse = false;
  };
  std::optional<Demotion> holdDemotion(const FactSet &OwnSet,
                                       const FactSet &OtherSet,
                                       const FactEntry &FE,
                                       LockErrorKind LEK) const {
    // Either form of the rebranch exemption: the hold's own proof in
    // \p OwnSet, or the inverse hold the rebranched call gave up.
    const TryFactEntry *Proof = rebranchProof(OwnSet, FE);
    if (Proof || isRebranchedInverseHold(OtherSet, FE)) {
      if (rebranchVetoedByReleased(OtherSet, FE))
        return std::nullopt;
      // The demoted try fact is the proving call's (of two merged identical
      // calls, whichever proved this side's hold); an inverse hold has no
      // proof of its own and takes the rebranched call. A rebranch that
      // does not resolve the result on every outgoing path still demotes,
      // and is reported there (joinDefiniteFromExit()): a branch that names
      // the call is the one that should have resolved it, so a same-origin
      // proof beside it does not make the silence right.
      return Demotion{Proof ? Proof->origin() : Ctx.RebranchTryLock,
                      /*Rebranch=*/true, /*Inverse=*/!Proof};
    }
    const TryFactEntry *W = sameOriginProof(OwnSet, OtherSet, FE);
    if (!W || !sameOriginGate(LEK, failureRecord(OtherSet, FE, W->origin())))
      return std::nullopt;
    return Demotion{W->origin(), /*Rebranch=*/false,
                    /*Inverse=*/W->negative() != FE.negative()};
  }
  /// Whether \p Set carries the Released try fact of \p Cap from \p Call:
  /// the call's stored result is stale there.
  bool releasedFor(const FactSet &Set, const CapabilityExpr &Cap,
                   const Expr *Call) const {
    return Set.findTryFactFrom(FactMan, Cap, Call,
                               TryFactEntry::State::Released);
  }
  /// The proof a forgiven mixed join rests on: the ProvedHeld try fact of
  /// \p DefSide in \p DefSet whose call \p CondSet carries the conditional
  /// try fact of, if the join is forgiven at all. The demotion adopts that
  /// call.
  const TryFactEntry *mixedJoinExempt(const FactSet &DefSet,
                                      const FactEntry &DefSide,
                                      const FactSet &CondSet) const;
  bool conditionalKeptAgainst(const TryFactEntry &W, const FactSet &OtherSet,
                              bool AllowRebranch = true) const;
  /// The conditional try fact of \p Cap from \p Origin in \p Kind in
  /// \p Set, if any.
  const TryFactEntry *condTryFact(const FactSet &Set, const CapabilityExpr &Cap,
                                  const Expr *Origin, LockKind Kind) const {
    const TryFactEntry *W = Set.findTryFact(FactMan, Cap, Origin, Kind);
    return W && W->conditional() ? W : nullptr;
  }
  /// \}

  /// \name Diagnostics
  /// \{
  void warnRemovedEntryFact(const FactEntry &EntryFact) const;
  void warnRemovedExitFact(const FactEntry &ExitFact) const;
  void warnNeverChecked(const TryFactEntry &W, LockErrorKind LEK);
  void warnReentrancyMismatch(const FactEntry &FE, LockErrorKind LEK) const;
  static unsigned depth(const FactEntry *Def, bool HasCond);
  /// \}

  /// \name Rewriting the merged set
  /// \{
  const FactEntry *demoteToConditional(const FactEntry &Def, const Expr *Origin,
                                       LockErrorKind LEK);
  void installConditionalOf(const FactEntry &Def, const Expr *Origin);
  bool demoteDeeperOfPair(FactSet::iterator EntryIt, FactID Fact,
                          const FactEntry &EntryFact,
                          const FactEntry &ExitFact);
  void demoteExitFact(const FactEntry &Def, const Expr *Origin,
                      LockErrorKind LEK);
  void demoteEntryFactInPlace(const FactEntry &Def, const Expr *Origin,
                              LockErrorKind LEK);
  /// \}

  /// \name The joins, by form
  /// A fact of the exit set is paired with its counterpart in the entry
  /// set (a "pair"), or is one-sided ("from exit"); a fact of the entry
  /// set without a counterpart is one-sided the other way ("from entry").
  /// \{
  void joinTryFactPair(FactSet::iterator EntryIt, const TryFactEntry &ExitW);
  void joinTryFactFromExit(FactID Fact, const TryFactEntry &ExitW);
  void joinTryFactFromEntry(const TryFactEntry &EntryW);
  void joinDefinitePair(FactSet::iterator EntryIt, FactID Fact,
                        const FactEntry &ExitFact);
  void joinDefiniteFromExit(FactID Fact, const FactEntry &ExitFact);
  void joinDefiniteFromEntry(const FactEntry &EntryFact);
  void joinMixedPair(FactSet::iterator EntryIt, FactID Fact,
                     const FactEntry &EntryFact, const FactEntry &ExitFact,
                     bool ExitHasCond);
  void joinMixedFromExit(FactID Fact, const FactEntry &ExitFact,
                         bool ExitHasCond);
  void joinMixedFromEntry(const FactEntry &EntryFact, bool EntryHasCond);
  /// \}
};

} // namespace

// The rebranched call's ProvedHeld try fact beside the hold \p Def in
// \p Set, the side \p Def belongs to: the hold was proved by the call the
// terminator branches on, so the edges can re-resolve it.
const TryFactEntry *LocksetJoin::rebranchProof(const FactSet &Set,
                                               const FactEntry &Def) const {
  if (!Ctx.RebranchTryLock)
    return nullptr;
  return Set.findTryFactIf(FactMan, Def, [&](const TryFactEntry &W) {
    return W.provedHeld() && isRebranchedCall(W.origin());
  });
}

// A definite hold of the capability a release-style try-acquire gives up
// (try_acquire_capability(true, !mu), this fact is mu) is one-sided at
// the join when the call's success released it on the other side -- and
// only then: the hold carries no proof of its own, so the evidence must
// be the call's own ProvedHeld try fact for the inverse capability, beside
// the negative fact it proved, in \p OtherSet. A hold that is one-sided
// for an unrelated reason (never acquired on that path) is diagnosed
// normally.
bool LocksetJoin::isRebranchedInverseHold(const FactSet &OtherSet,
                                          const FactEntry &FE) const {
  if (!Ctx.RebranchTryLock || FE.negative())
    return false;
  const FactEntry *Released = OtherSet.findDefinite(FactMan, !FE);
  return Released && rebranchProof(OtherSet, *Released);
}

// A one-sided hold under the rebranch exemption is carried (demoted to
// the call's conditional try fact) on the premise that the call's stored
// result is falsy on the side missing it: lost to the call's failure
// edge, or never acquired. That call's Released try fact on the other side
// refutes the premise: some path there keeps a truthy stored result for
// a hold that is gone, so re-resolving the carried try fact could
// resurrect the dead hold -- e.g. the release in
// `if (c) { if (ok) mu.Unlock(); }` followed by another `if (ok)`.
// Another call's release says nothing about this call's result: the veto
// is per call.
bool LocksetJoin::rebranchVetoedByReleased(const FactSet &OtherSet,
                                           const FactEntry &FE) const {
  return OtherSet.findTryFactIf(FactMan, FE, [&](const TryFactEntry &W) {
    return W.released() && isRebranchedCall(W.origin());
  });
}

// Whether a conditional try fact one-sided at the join is carried into the
// merged state: whenever the other side holds the capability in any form
// (the other side's extra is what gets diagnosed), or the terminator
// rebranches on its call. Missing from a path that does not hold the
// capability at all, the analysis loses track of it.
bool LocksetJoin::conditionalKeptAgainst(const TryFactEntry &W,
                                         const FactSet &OtherSet,
                                         bool AllowRebranch) const {
  return OtherSet.findDefinite(FactMan, W) ||
         OtherSet.anyConditional(FactMan, W) ||
         (AllowRebranch && isTrylockRebranched(W));
}

// The same-origin analogue of the rebranch exemption for a one-sided
// hold: the join's other side carries the failure record of a call that
// proved the hold, so the sides are exactly "held iff C's result" and
// "C's result is falsy", and their join is the call's conditional try fact.
// The other side may instead carry the call's own proved release of the
// capability (a hold the call gives up on success, proved by the call's
// failure): the sides are then "held iff C failed" and "C succeeded", and
// their join is the same conditional try fact, which resolves inverted
// (getEdgeLockset()). Returns the proving try fact, whose call the
// demotion adopts.
//
// The failure record alone settles the other side: the caller establishes
// that the other side does not hold the level being demoted (it has no
// definite fact of the capability at all, or one level less deep), so C's
// ProvedNotHeld try fact there says that level is not held. Asking for the
// negative fact instead would refuse the demotion exactly where a second
// try-acquire of the capability is in flight -- the failure edge installs
// no negative fact while another call's conditional try fact survives --
// and those conditionals are what carries the possible hold, not the
// demoted level.
const TryFactEntry *LocksetJoin::sameOriginProof(const FactSet &OwnSet,
                                                 const FactSet &OtherSet,
                                                 const FactEntry &FE) const {
  if (const TryFactEntry *W =
          OwnSet.findTryFactIf(FactMan, FE, [&](const TryFactEntry &W) {
            if (!W.provedHeld())
              return false;
            const Expr *C = W.origin();
            if (OtherSet.findTryFactFrom(FactMan, FE, C,
                                         TryFactEntry::State::ProvedNotHeld))
              return true;
            // The inverse form, where this side also proved the hold: the
            // other side's release needs its negative fact beside the proof.
            return OtherSet.findTryFactFrom(FactMan, !FE, C,
                                            TryFactEntry::State::ProvedHeld) &&
                   OtherSet.findDefinite(FactMan, !FE);
          }))
    return W;
  // The inverse form proper: what C proved is the release, so this side's
  // hold carries no try fact of its own and the proof is the other side's,
  // keyed on the negative capability. This side records the same call's
  // failure there -- nothing was released -- which is what makes the two
  // sides one call's two outcomes. The other side need not hold the
  // negative capability definitely: a partial release of a reentrant hold
  // leaves the capability held, one level shallower, which is precisely
  // the level the demotion conditions on C.
  return OtherSet.findTryFactIf(FactMan, !FE, [&](const TryFactEntry &W) {
    return W.provedHeld() &&
           OwnSet.findTryFactFrom(FactMan, !FE, W.origin(),
                                  TryFactEntry::State::ProvedNotHeld);
  });
}

// Where the silent same-origin reconstitution applies: at loop joins
// always; at branch joins where the hidden leak stays diagnosable or was
// never diagnosed -- under -Wthread-safety-beta (the unchecked-result
// diagnostics report it downstream) or for a `?:` arm split (\p NotHeldW
// carries the mark: the branch was never honored before, so its join
// never warned). Otherwise the eager lost-hold diagnosis at the join is
// the only coverage and is retained. Naming both join kinds also keeps
// the exemption away from the end-of-function comparison, whose entry set
// is the declared expected set: rewriting that would swallow the
// still-held diagnostic. (No try fact there today, so this only pins the
// invariant.)
bool LocksetJoin::sameOriginGate(LockErrorKind LEK,
                                 const TryFactEntry *NotHeldW) const {
  return LEK == LEK_LockedSomeLoopIterations ||
         (LEK == LEK_LockedSomePredecessors &&
          (Handler.issueBetaWarnings() || (NotHeldW && NotHeldW->armSplit())));
}

// The same-origin mixed join: a definite hold proved by call C meets C's
// own conditional try fact on the other side. Both sides' states are
// conditioned on C's result -- the hold is merely edge-strengthened by a
// check of a previous execution of C -- so the mixed join loses nothing
// and the merged state is the conditional one: always at loop joins (the
// check-first loop idioms `if (ok) continue; ok = mu.TryLock();` create
// this shape at continue latches), unless C's hold was released on either
// side, which refutes the shared condition; at branch joins only through
// the terminator rebranching on C, which re-resolves the demoted state
// on its outgoing edges. Under a rebranch on the merged result of two
// identical calls, the other side's conditional try fact may be the twin
// call's: the retry-once idiom `if (!ok) ok = mu.TryLock();` meets the
// first call's proved hold with the second call's conditional try fact,
// and the branch on the merged variable resolves either.
const TryFactEntry *LocksetJoin::mixedJoinExempt(const FactSet &DefSet,
                                                 const FactEntry &DefSide,
                                                 const FactSet &CondSet) const {
  return DefSet.findTryFactIf(FactMan, DefSide, [&](const TryFactEntry &W) {
    if (!W.provedHeld())
      return false;
    const LockKind Kind = DefSide.kind();
    if (isRebranchedCall(W.origin()) && Ctx.RebranchResolvesAllPaths)
      return condTryFact(CondSet, DefSide, Ctx.RebranchTryLock, Kind) ||
             (Ctx.RebranchMergedCall &&
              condTryFact(CondSet, DefSide, Ctx.RebranchMergedCall, Kind));
    return condTryFact(CondSet, DefSide, W.origin(), Kind) &&
           Ctx.isLoopJoin() &&
           !releasedFor(EntrySetOrig, DefSide, W.origin()) &&
           !releasedFor(ExitSet, DefSide, W.origin());
  });
}

// Warn about a fact the intersection removes (or weakens to conditional).
// However, a capability managed by a scoped object is exempt -- the
// scoped fact still knows to release it -- except where the scope itself
// ends or repeats.
void LocksetJoin::warnRemovedEntryFact(const FactEntry &EntryFact) const {
  if (!EntryFact.managed() || Ctx.ExitLEK == LEK_LockedSomeLoopIterations ||
      Ctx.ExitLEK == LEK_NotLockedAtEndOfFunction)
    EntryFact.handleRemovalFromIntersection(EntrySetOrig, FactMan, Ctx.JoinLoc,
                                            Ctx.ExitLEK, Handler);
}

void LocksetJoin::warnRemovedExitFact(const FactEntry &ExitFact) const {
  if (!ExitFact.managed() || Ctx.EntryLEK == LEK_LockedAtEndOfFunction)
    ExitFact.handleRemovalFromIntersection(ExitSet, FactMan, Ctx.JoinLoc,
                                           Ctx.EntryLEK, Handler);
}

// Likewise for the beta diagnostic that a try-acquire's possible success
// is carried into the join (or out of the function) unchecked: for a
// conditional try fact the analysis loses track of. Emitted once per (join,
// acquisition, capability): the same unchecked result can be lost at more
// than one join -- by the pairwise intersection of a many-predecessor join,
// or on two different paths out of the acquisition -- and the leak it
// reports is one, so it is reported at the first join that loses it
// (NeverCheckedWarned).
// \p LEK is the error kind governing the branch that fires the warning
// (EntryLEK for a try fact from the exit set, ExitLEK for one from the
// entry set); it decides the at-end-of-function wording.
void LocksetJoin::warnNeverChecked(const TryFactEntry &W, LockErrorKind LEK) {
  assert(W.conditional() &&
         "only a conditional try fact carries an unchecked result");
  // Recorded whether or not the diagnostic is enabled: what a join loses
  // here is a result nothing accounts for, which the merge of two calls'
  // results must not paper over (FactManager::lostUnchecked()).
  FactMan.addLostUnchecked(W.origin());
  if (!Handler.issueBetaWarnings() || !W.lossNeedsWarning())
    return;
  // A capability managed by a scoped object is exempt at an interior
  // join, as for the lost-hold diagnostics above: the scoped fact is
  // still live and its destructor discharges the conditional acquisition.
  // Only at the end of the function, where the guard may have escaped its
  // scope, can the guard fail to check it. (A loop join reaches here now
  // that the exemption there is narrowed to the results the loop really
  // checks -- the caller's decision still, joinTryFactFromExit().)
  if (W.managed() && !isEndOfFunctionLEK(LEK))
    return;
  if (!Analyzer.NeverCheckedWarned
           .insert({W.origin(), W.sexpr(),
                    2 * unsigned(W.kind()) + unsigned(W.negative())})
           .second)
    return;
  SourceLocation LocAcquired = W.origin()->getExprLoc();
  std::string Name = W.toString();
  Handler.handleTryAcquireNeverChecked(W.getKind(), Name, LocAcquired,
                                       Ctx.JoinLoc,
                                       /*AtEndOfFunction=*/
                                       isEndOfFunctionLEK(LEK));
}

// Diagnose a join where only the guaranteed depth of the hold differs.
void LocksetJoin::warnReentrancyMismatch(const FactEntry &FE,
                                         LockErrorKind LEK) const {
  // Only a reentrant capability has levels to disagree about. On any other,
  // a level that exists only conditionally is the modeling of a try-acquire
  // over a hold (ThreadSafetyAnalyzer::addTryLock()), which at runtime fails
  // rather than nesting: both sides hold the capability, so there is nothing
  // to report, and "reentrancy depth" would name something it does not have.
  // A level a branch proved is a different matter, and the definite join
  // still reports losing it (LockableFactEntry::handleRemovalFromIntersection).
  if (!FE.reentrant())
    return;
  Handler.handleMutexHeldEndOfScope(FE.getKind(), FE.toString(), FE.loc(),
                                    Ctx.JoinLoc, LEK,
                                    /*ReentrancyMismatch=*/true);
}

// The total number of levels a side holds, definite or conditional: what
// a reentrancy-depth comparison of the two sides sees. A ProvedHeld
// try fact adds nothing: its level is the definite fact's.
unsigned LocksetJoin::depth(const FactEntry *Def, bool HasCond) {
  return (Def ? reentrancyDepth(*Def) + 1 : 0) + (HasCond ? 1 : 0);
}

// Demote the definite hold \p Def, proved by the try-acquire call
// \p Origin (or given up by the rebranched one: a hold the call releases
// on success carries no proof of its own, and the demotion adopts the
// call, whose outcome resolves the hold inverted, getEdgeLockset()), to
// that call's conditional try fact in the entry set, which a branch on the
// result resolves. A mismatched reentrancy depth is diagnosed here but
// kept -- after the warning, the deeper fact guards more of the releases
// downstream than a stripped one would -- as the levels below the demoted
// one, no longer proved by any call: returned for the caller to place,
// since \p Def itself may belong to the other side.
const FactEntry *LocksetJoin::demoteToConditional(const FactEntry &Def,
                                                  const Expr *Origin,
                                                  LockErrorKind LEK) {
  const auto &LDef = cast<LockableFactEntry>(Def);
  if (LDef.getReentrancyDepth() != 0)
    warnReentrancyMismatch(Def, LEK);
  installConditionalOf(Def, Origin);
  return LDef.leaveReentrant(FactMan);
}

// The merged set's record that \p Def's top level is held only if
// \p Origin succeeded: that call's try fact, turned conditional where the
// set already resolved it, added from \p Def where the set has none. The
// proof the call gave on its own side does not hold on the other.
void LocksetJoin::installConditionalOf(const FactEntry &Def,
                                       const Expr *Origin) {
  if (FactSet::iterator It =
          EntrySet.findTryFactIter(FactMan, Def, Origin, Def.kind());
      It != EntrySet.end()) {
    const auto &W = cast<TryFactEntry>(FactMan[*It]);
    if (!W.conditional())
      EntrySet.replaceFact(
          FactMan, It, W.withState(FactMan, TryFactEntry::State::Conditional));
    return;
  }
  EntrySet.addLock(FactMan,
                   cast<LockableFactEntry>(Def).asConditional(FactMan, Origin));
}

// Demote the exit set's hold \p Def into the entry set, adding the levels
// below it beside the conditional try fact.
void LocksetJoin::demoteExitFact(const FactEntry &Def, const Expr *Origin,
                                 LockErrorKind LEK) {
  if (const FactEntry *Rest = demoteToConditional(Def, Origin, LEK))
    EntrySet.addLock(FactMan, Rest);
}

// Demote the entry set's own hold \p Def in place: the levels below it
// take its slot, or it is removed.
void LocksetJoin::demoteEntryFactInPlace(const FactEntry &Def,
                                         const Expr *Origin,
                                         LockErrorKind LEK) {
  if (const FactEntry *Rest = demoteToConditional(Def, Origin, LEK))
    EntrySet.replaceFact(FactMan, Def, Rest);
  else
    EntrySet.removeFact(FactMan, Def);
}

// Two try facts of one identity: the merged state is their join, except where
// the lattice's same-origin cells meet a policy.
//
// A conditional try fact meeting the same call's Released try fact loses its
// possible hold to the stale result: the capability may be held on its
// side with the result never checked there, diagnosed as for a one-sided
// conditional try fact (joinTryFactFromExit()) unless the other side holds
// the capability itself -- a rebranch does not save it, the stale result
// vetoes carrying it.
//
// A ProvedNotHeld try fact meeting the call's conditional or ProvedHeld try
// fact is the same-origin join: "held iff C" (or "held, proved by C") meets "C
// is falsy", which is "held iff C" again. That reconstitution is not taken
// here: the conditional side's possible hold is lost -- unless the other side
// holds the capability itself -- and a proved hold is diagnosed as lost by the
// definite join, leaving nothing for a try fact to say.
void LocksetJoin::joinTryFactPair(FactSet::iterator EntryIt,
                                  const TryFactEntry &ExitW) {
  using State = TryFactEntry::State;
  const auto &EntryW = cast<TryFactEntry>(FactMan[*EntryIt]);
  if (EntryW.state() == ExitW.state()) {
    // The state agrees, but the marks may not: release evidence on either
    // side is release evidence at the join (a stale result is stale however
    // it got here), so the mark is a union rather than the entry side's.
    if (EntryW.conditional() && ExitW.mayBeReleased() &&
        !EntryW.mayBeReleased() && Ctx.canModify())
      EntrySet.replaceFact(FactMan, EntryW,
                           EntryW.asMayBeReleased(FactMan, ExitW.releaseLoc()));
    // The `?:` arm-split mark is an annotation on one edge, not a state:
    // it is spent at the join of the arms it marks, so a merged fact never
    // carries it. Keeping the entry side's object mark-and-all made the
    // exemption follow whichever arm the walk reached first.
    else if (EntryW.armSplit() && Ctx.canModify())
      EntrySet.replaceFact(FactMan, EntryW,
                           EntryW.withState(FactMan, EntryW.state()));
    return;
  }
  State Merged = TryFactEntry::joinStates(EntryW.state(), ExitW.state());
  const TryFactEntry *CondSide = EntryW.conditional()  ? &EntryW
                                 : ExitW.conditional() ? &ExitW
                                                       : nullptr;
  const FactSet &CondOther = CondSide == &EntryW ? ExitSet : EntrySetOrig;
  const LockErrorKind CondLEK =
      CondSide == &EntryW ? Ctx.ExitLEK : Ctx.EntryLEK;
  // A Conditional try fact meeting a Released one: the conditional side keeps
  // its possible hold, marked MayBeReleased, only where the join would keep it
  // one-sided -- not under a rebranch, whose resolution the stale result
  // vetoes -- and loses it, diagnosed, otherwise; released on either side is
  // released in the merged set wherever the join keeps such evidence at all,
  // a continue latch included, so that a release inside the loop body
  // reaches the post-loop resolution.
  if (Merged == State::Released) {
    if (CondSide && Ctx.canModify()) {
      if (conditionalKeptAgainst(*CondSide, CondOther,
                                 /*AllowRebranch=*/false)) {
        if (!EntryW.conditional() || !EntryW.mayBeReleased()) {
          const TryFactEntry &ReleasedSide =
              CondSide == &EntryW ? ExitW : EntryW;
          EntrySet.replaceFact(FactMan, EntryIt,
                               CondSide->mayBeReleased()
                                   ? CondSide
                                   : CondSide->asMayBeReleased(
                                         FactMan, ReleasedSide.releaseLoc()));
        }
        return;
      }
      warnNeverChecked(*CondSide, CondLEK);
    }
    // Below runs at a continue latch too, where the block above did not:
    // taking the released side there replaces a conditional side's possible
    // hold without the report, which is the loop exemption doing its work
    // (a result the code checks on the paths around the loop is not a leak,
    // Ctx.isLoopJoin()), not a report the arm above dropped.
    // This is the branch's own join when the other side is its failure
    // edge, and a stale result no later branch can name has then done its
    // work: dropping it is what keeps a long run of self-contained
    // `if (mu.TryLock()) { ...; mu.Unlock(); }` from carrying one released
    // record per statement to the end of the function, where the pairwise
    // joins pay for all of them.
    if (!CondSide && Ctx.canModify() &&
        (EntryW.provedNotHeld() || ExitW.provedNotHeld()) &&
        !resultCanBeRebranched(EntryW)) {
      EntrySet.erase(EntryIt);
      return;
    }
    // The released side's try fact carries the release location.
    if (Ctx.mayKeepReleasedTryFact() && !EntryW.released())
      EntrySet.replaceFact(FactMan, EntryIt, &ExitW);
    return;
  }
  if (EntryW.provedNotHeld() || ExitW.provedNotHeld()) {
    // The same-origin reconstitution: allowed under the policy gate when
    // the failure record comes with its side's negative fact, the
    // merged try fact is conditional (a proved hold on the other side is
    // demoted silently by the definite join, whose exemption agrees).
    const bool EntryNotHeld = EntryW.provedNotHeld();
    const TryFactEntry &NotHeldW = EntryNotHeld ? EntryW : ExitW;
    const FactEntry *NotHeldNeg =
        (EntryNotHeld ? EntrySetOrig : ExitSet).findDefinite(FactMan, !ExitW);
    const bool Allowed = NotHeldNeg && sameOriginGate(Ctx.EntryLEK, &NotHeldW);
    if (Ctx.isLoopJoin()) {
      // A sealed loop join leaves the entry set alone; a continue latch
      // takes the reconstitution.
      if (Ctx.isUnsealedLoopJoin() && Allowed && EntryNotHeld)
        EntrySet.replaceFact(
            FactMan, EntryIt,
            CondSide ? CondSide
                     : EntryW.withState(FactMan, State::Conditional));
      return;
    }
    if (Allowed || (CondSide && conditionalKeptAgainst(*CondSide, CondOther))) {
      // The conditional side, may-be-released or not, is the merged try fact.
      if (CondSide ? CondSide != &EntryW : !EntryW.conditional())
        EntrySet.replaceFact(
            FactMan, EntryIt,
            CondSide ? CondSide
                     : EntryW.withState(FactMan, State::Conditional));
      return;
    }
    if (CondSide)
      warnNeverChecked(*CondSide, CondLEK);
    // Nothing is left to record -- except the stale result of a
    // conditional side marked MayBeReleased, kept as Released like the evidence
    // it is.
    if (CondSide && CondSide->mayBeReleased() && Ctx.mayKeepReleasedTryFact())
      EntrySet.replaceFact(FactMan, EntryIt,
                           CondSide->withState(FactMan, State::Released));
    else
      EntrySet.erase(EntryIt);
    return;
  }
  // Conditional meeting ProvedHeld (the mixed join): the conditional side,
  // may-be-released or not, is the merged try fact; the definite join diagnoses
  // or demotes the proved hold. A sealed loop join leaves the entry set alone;
  // a continue latch takes the same-origin mixed exemption.
  if (Ctx.isLoopJoin()) {
    if (Ctx.isUnsealedLoopJoin() && EntryW.provedHeld()) {
      const FactEntry *EntryDef = EntrySetOrig.findDefinite(FactMan, EntryW);
      if (EntryDef && mixedJoinExempt(EntrySetOrig, *EntryDef, ExitSet))
        EntrySet.replaceFact(FactMan, EntryIt, CondSide);
    }
    return;
  }
  if (CondSide != &EntryW)
    EntrySet.replaceFact(FactMan, EntryIt, CondSide);
}

// A try fact of the exit set without a counterpart. A conditional one is
// carried into the merged state whenever the other side holds the capability in
// any form (the other side's extra is what gets diagnosed), or the terminator
// rebranches on its call. Missing from a path that does not hold the capability
// at all, the analysis loses track of it: this predecessor carries a
// try-acquire result into the join without its result having been checked --
// the capability may be leaked, the beta diagnostic. So does one reaching the
// end of the function, however the expected set holds the capability: nothing
// after can check it. At a loop join it is a leak only when the result is not
// branched on anywhere inside the loop (JoinContext::CheckedAroundLoop): the
// next iteration re-executes the call, or the loop discards the result,
// unchecked. A ProvedHeld try fact is never carried one-sided: the merged hold,
// if any, is not proved by its call, and the definite join demotes it to
// conditional where the rebranch exemption applies.
void LocksetJoin::joinTryFactFromExit(FactID Fact, const TryFactEntry &ExitW) {
  if (ExitW.released()) {
    // A stale result on this predecessor only: kept as evidence -- not
    // held on some path -- at the forward joins that may grow the entry
    // set, for the rebranch and re-materialization vetoes.
    if (Ctx.mayKeepReleasedTryFact())
      EntrySet.addLockByID(Fact);
    return;
  }
  // A ProvedHeld try fact is never carried one-sided (above), and a
  // ProvedNotHeld one proves the failure on its own side only: dropped.
  if (!ExitW.conditional())
    return;
  if (Ctx.isLoopJoin()) {
    // At a loop join it warns only when the result is not branched on
    // anywhere inside the loop (CheckedAroundLoop): then the next
    // iteration re-executes the call (or the loop discards the result)
    // while this iteration's possible success was never checked -- a
    // check after the loop sees only the last result and cannot make
    // this sound. Joins without that information (continue joins, see
    // runAnalysis()) stay exempt, and so does a try fact the pre-loop
    // state holds definitely (the depth mismatch is diagnosed instead);
    // another call's conditional try fact there is no check of this one.
    if (!EntrySetOrig.findDefinite(FactMan, ExitW) &&
        !isTrylockRebranched(ExitW) && Ctx.CheckedAroundLoop &&
        !Ctx.CheckedAroundLoop->count(ExitW.origin()))
      warnNeverChecked(ExitW, Ctx.EntryLEK);
    return;
  }
  if (Ctx.EntryLEK == LEK_LockedAtEndOfFunction) {
    warnNeverChecked(ExitW, Ctx.EntryLEK);
    return;
  }
  // Two structurally identical calls whose merged result the joining
  // block's terminator branches on (RebranchMergedCall): a side holding one
  // of them alone holds "the result of that call" either way, so its
  // try fact folds into the try fact of the call the merge resolves to (the
  // first path's), which the outgoing edges resolve like a single call's.
  // A side holding both executed the second call over the first's
  // unresolved try fact: the variable then holds only the second call's
  // result, and the first's try fact is left alone, to be reported
  // unchecked where it is lost.
  const Expr *Rebranch = Ctx.RebranchTryLock, *Twin = Ctx.RebranchMergedCall;
  const LockKind Kind = ExitW.kind();
  if (Twin && ExitW.origin() == Twin &&
      !condTryFact(ExitSet, ExitW, Rebranch, Kind) &&
      condTryFact(EntrySetOrig, ExitW, Rebranch, Kind))
    return;
  if (Twin && ExitW.origin() == Rebranch &&
      !condTryFact(ExitSet, ExitW, Twin, Kind) &&
      !condTryFact(EntrySetOrig, ExitW, Rebranch, Kind))
    if (const TryFactEntry *TwinW =
            condTryFact(EntrySetOrig, ExitW, Twin, Kind))
      EntrySet.removeFact(FactMan, *TwinW);
  if (conditionalKeptAgainst(ExitW, EntrySetOrig)) {
    EntrySet.addLockByID(Fact);
    return;
  }
  warnNeverChecked(ExitW, Ctx.EntryLEK);
  // A lost may-be-released try fact leaves its stale result behind as evidence,
  // where the join keeps such evidence.
  if (ExitW.mayBeReleased() && Ctx.mayKeepReleasedTryFact())
    EntrySet.addLock(FactMan,
                     ExitW.withState(FactMan, TryFactEntry::State::Released));
}

// A try fact of the entry set without a counterpart: as above, a conditional
// one is kept wherever the other side holds the capability in any form or the
// terminator rebranches on its call, and lost otherwise -- with the unchecked
// try-acquire on an earlier predecessor; the beta warning again not at a loop
// join (a conditionally held try fact missing from a loop's back edge was
// checked inside the loop, which is not a leak). A one-sided ProvedHeld try
// fact proves nothing on the other path and is dropped (unless the definite
// join already turned it conditional, in which case it is no longer here to
// drop).
void LocksetJoin::joinTryFactFromEntry(const TryFactEntry &EntryW) {
  // As above, a stale result stays as evidence wherever the join may keep
  // it, which every join that rewrites the entry set does.
  if (EntryW.released())
    return;
  if (EntryW.conditional()) {
    // A try fact of the second of two identical calls the terminator's
    // merged variable resolves (RebranchMergedCall), held alone on this
    // side, folds into the first call's try fact, kept from the exit side
    // above.
    const Expr *Rebranch = Ctx.RebranchTryLock, *Twin = Ctx.RebranchMergedCall;
    const LockKind Kind = EntryW.kind();
    if (Twin && Ctx.ExitLEK == LEK_LockedSomePredecessors &&
        EntryW.origin() == Twin &&
        !condTryFact(EntrySetOrig, EntryW, Rebranch, Kind) &&
        condTryFact(ExitSet, EntryW, Rebranch, Kind)) {
      EntrySet.removeFact(FactMan, EntryW);
      return;
    }
    if (conditionalKeptAgainst(EntryW, ExitSet))
      return;
    // (A try fact the other side records the failure of is a pair, joined
    // above under the same-origin exemption.)
    // (The CheckedAroundLoop narrowing the exit-set side applies is
    // deliberately not mirrored here: a try fact reaching this arm was
    // released inside the loop, which is diagnosed at the release itself
    // unless an assert claimed the hold -- and a loop join leaves the
    // entry set unmodified, so the try fact is diagnosed again wherever it
    // is finally lost.)
    if (Ctx.ExitLEK != LEK_LockedSomeLoopIterations)
      warnNeverChecked(EntryW, Ctx.ExitLEK);
    // As above: a lost may-be-released try fact leaves its stale result
    // behind, wherever the join keeps such evidence at all -- the same
    // question joinTryFactFromExit() asks, and a property of the join, not
    // of which side is missing the fact.
    if (EntryW.mayBeReleased() && Ctx.mayKeepReleasedTryFact()) {
      EntrySet.replaceFact(
          FactMan, EntryW,
          EntryW.withState(FactMan, TryFactEntry::State::Released));
      return;
    }
  }
  // Only a branch join rewrites the entry set.
  if (Ctx.ExitLEK == LEK_LockedSomePredecessors)
    EntrySet.removeFact(FactMan, EntryW);
}

// Two definite facts of a capability: a mixed join if exactly one side also
// holds it conditionally, else joined. A ProvedHeld try fact on one side only
// is dropped by the try fact joins: the merged hold is not proved by that call.
void LocksetJoin::joinDefinitePair(FactSet::iterator EntryIt, FactID Fact,
                                   const FactEntry &ExitFact) {
  const FactEntry &EntryFact = FactMan[*EntryIt];
  if (ExitFact.negative()) {
    // Two negative facts: joined as ever. Their try facts -- a try-release
    // proved on one side, a failure or release recorded on one side -- join
    // through the try fact lattice like any other.
    if (Analyzer.join(EntryFact, ExitFact, Ctx.JoinLoc, Ctx.EntryLEK))
      *EntryIt = Fact;
    return;
  }
  const bool ExitHasCond = ExitSet.anyConditional(FactMan, ExitFact);
  const bool EntryHasCond = EntrySetOrig.anyConditional(FactMan, ExitFact);
  if (EntryHasCond != ExitHasCond)
    joinMixedPair(EntryIt, Fact, EntryFact, ExitFact, ExitHasCond);
  else if (!demoteDeeperOfPair(EntryIt, Fact, EntryFact, ExitFact) &&
           Analyzer.join(EntryFact, ExitFact, Ctx.JoinLoc, Ctx.EntryLEK))
    *EntryIt = Fact;
}

// Two definite holds one level apart, where the deeper side's extra level
// is exactly what its own try-acquire proved and the other side settles
// that call's result the other way (holdDemotion()): the sides do not
// disagree about a depth, they hold one more level if the call succeeded.
// The merged state is the shallower hold with the call's conditional try
// fact beside it, which a branch on the result resolves -- the definite
// join's analogue of the demotion the one-sided paths apply, silent for
// the same reason. One level only: a wider gap is a disagreement no single
// call accounts for, and the depth mismatch is reported as ever.
bool LocksetJoin::demoteDeeperOfPair(FactSet::iterator EntryIt, FactID Fact,
                                     const FactEntry &EntryFact,
                                     const FactEntry &ExitFact) {
  if (!Ctx.canModify())
    return false;
  const unsigned EntryDepth = reentrancyDepth(EntryFact);
  const unsigned ExitDepth = reentrancyDepth(ExitFact);
  const bool ExitDeeper = ExitDepth == EntryDepth + 1;
  if (!ExitDeeper && EntryDepth != ExitDepth + 1)
    return false;
  const FactEntry &Deep = ExitDeeper ? ExitFact : EntryFact;
  const FactSet &DeepSet = ExitDeeper ? ExitSet : EntrySetOrig;
  const FactSet &Other = ExitDeeper ? EntrySetOrig : ExitSet;
  const LockErrorKind LEK = ExitDeeper ? Ctx.EntryLEK : Ctx.ExitLEK;
  std::optional<Demotion> D = holdDemotion(DeepSet, Other, Deep, LEK);
  // A rebranch behind a short circuit leaves an edge the result does not
  // resolve, as for a one-sided hold: the level is diagnosed there after
  // all, which the ordinary join does.
  if (!D || (D->Rebranch && !Ctx.RebranchResolvesAllPaths))
    return false;
  if (D->Inverse) {
    // What the call proved is the release, so the deeper side is the one
    // where it failed and the merged state is that deeper hold: the level
    // the call may have discharged rides on the call's own conditional try
    // fact of the negative capability, which joinTryFactPair()
    // reconstitutes beside it and a later branch discharges again.
    if (ExitDeeper)
      *EntryIt = Fact;
    return true;
  }
  if (!ExitDeeper)
    *EntryIt = Fact;
  installConditionalOf(Deep, D->Origin);
  return true;
}

// Mixed, both sides definite: one side's hold is one conditional level deeper.
// Forgiven when the definite side's extra level was proved by the call the
// terminator rebranches on and the other side holds that call's conditional try
// fact: the merged state re-resolves it. Otherwise the capability is held on
// every path and only the guaranteed depth differs: the reentrancy-mismatch
// wording, like a join of unequal definite depths, under the exemptions of the
// lost-hold path it replaces (a scoped object still knows to release the levels
// it manages at an interior join). The merged state is the conditional side's.
void LocksetJoin::joinMixedPair(FactSet::iterator EntryIt, FactID Fact,
                                const FactEntry &EntryFact,
                                const FactEntry &ExitFact, bool ExitHasCond) {
  const FactEntry &DefSide = ExitHasCond ? EntryFact : ExitFact;
  const FactEntry &CondSide = ExitHasCond ? ExitFact : EntryFact;
  const FactSet &DefSet = ExitHasCond ? EntrySetOrig : ExitSet;
  const FactSet &CondSet = ExitHasCond ? ExitSet : EntrySetOrig;
  const TryFactEntry *Proof = mixedJoinExempt(DefSet, DefSide, CondSet);
  if (!Proof) {
    if (CondSide.lossNeedsWarning() &&
        !(CondSide.managed() && Ctx.EntryLEK == LEK_LockedSomePredecessors))
      warnReentrancyMismatch(CondSide, Ctx.EntryLEK);
  } else if (Ctx.canModify() &&
             depth(&EntryFact, !ExitHasCond) != depth(&ExitFact, ExitHasCond)) {
    // A forgiven mixed join with unequal reentrancy depths is otherwise
    // silent: diagnose the mismatch here.
    warnReentrancyMismatch(ExitFact, Ctx.EntryLEK);
  } else if (Ctx.isUnsealedLoopJoin() && ExitHasCond) {
    // Under the same-origin exemption the merged state must be the
    // conditional one: the continue latch keeps the entry's definite
    // fact, which the exit side's try fact demotes (the conditional
    // try fact itself was carried by joinTryFactPair(); reentrancy depth
    // loss is still diagnosed). By identity, not through EntryIt: the
    // demotion may add to the set.
    demoteEntryFactInPlace(EntryFact, Proof->origin(), Ctx.EntryLEK);
    return;
  }
  if (Ctx.canModify() && ExitHasCond)
    *EntryIt = Fact;
}

// A definite hold of this predecessor only: a mixed join if the other side
// holds the capability conditionally; else demoted to conditional under the
// rebranch exemption; else lost, the default lost-capability diagnostic (its
// conditional try facts, if any, are lost by joinTryFactFromExit()).
void LocksetJoin::joinDefiniteFromExit(FactID Fact, const FactEntry &ExitFact) {
  // A negative fact on this predecessor only leaves the intersection, as
  // any one-sided fact does: release evidence for the conditional-hold
  // machinery lives on the try facts (Released), not on negative facts. The
  // rebranched call's own proved release of a negative capability is that
  // call's resolved acquisition and takes the rebranch demotion below like any
  // proved hold.
  if (ExitFact.negative() && !rebranchProof(ExitSet, ExitFact))
    return;
  if (EntrySetOrig.anyConditional(FactMan, ExitFact)) {
    joinMixedFromExit(Fact, ExitFact,
                      ExitSet.anyConditional(FactMan, ExitFact));
    return;
  }
  if (std::optional<Demotion> D =
          holdDemotion(ExitSet, EntrySetOrig, ExitFact, Ctx.EntryLEK)) {
    // Held on this predecessor only, and either the terminator
    // rebranches on the try-acquire that proved it (or gives it up;
    // getEdgeLockset() will re-resolve it on the outgoing edges) or the
    // other side carries the call's own failure record: demote it to
    // conditionally held.
    if (!Ctx.canModify()) {
      // A sealed join cannot record the demotion: the entry set is the
      // one an earlier pass already analyzed the block with. The hold is
      // gone from the merged state either way, so it is reported like any
      // other lost hold rather than silently dropped.
      warnRemovedExitFact(ExitFact);
      return;
    }
    // A rebranch behind a short-circuit does not resolve the result on
    // the escaping edge: a definite hold weakened here can leak there,
    // so it is diagnosed at this join after all (the demotion stands,
    // for the paths that do rebranch). This covers both forms of the
    // rebranch exemption; the same-origin exemption is the one that
    // stays silent, since the other side carries the same call's own
    // failure record and the merged try fact is what a later branch on
    // the result resolves.
    if (D->Rebranch && !Ctx.RebranchResolvesAllPaths)
      warnRemovedExitFact(ExitFact);
    demoteExitFact(ExitFact, D->Origin, Ctx.EntryLEK);
    return;
  }
  // The hold is lost on the other path, whether or not the capability is
  // also conditionally held here: a possible hold beside it does not make a
  // certain one any less lost.
  warnRemovedExitFact(ExitFact);
}

// A definite hold of the entry set only: as above, but a demotion keeps the
// fact in the intersection in its demoted conditionally held form (except at a
// loop join, where the entry set is left unmodified).
void LocksetJoin::joinDefiniteFromEntry(const FactEntry &EntryFact) {
  if (EntryFact.negative() && !rebranchProof(EntrySetOrig, EntryFact)) {
    // As above: a one-sided negative leaves the intersection at a branch
    // join; other joins leave the entry set unmodified.
    if (Ctx.ExitLEK == LEK_LockedSomePredecessors)
      EntrySet.removeFact(FactMan, EntryFact);
    return;
  }
  if (ExitSet.anyConditional(FactMan, EntryFact)) {
    joinMixedFromEntry(EntryFact,
                       EntrySetOrig.anyConditional(FactMan, EntryFact));
    return;
  }
  if (std::optional<Demotion> D =
          holdDemotion(EntrySetOrig, ExitSet, EntryFact, Ctx.ExitLEK)) {
    if (!Ctx.canModify()) {
      // As above: a sealed join cannot record the demotion, so the lost
      // hold is reported.
      warnRemovedEntryFact(EntryFact);
      return;
    }
    // As above: an escaping short-circuit edge means the weakened
    // definite hold is diagnosed at the join after all, for both forms
    // of the rebranch exemption but not for the same-origin one.
    if (D->Rebranch && !Ctx.RebranchResolvesAllPaths)
      warnRemovedEntryFact(EntryFact);
    demoteEntryFactInPlace(EntryFact, D->Origin, Ctx.ExitLEK);
    return;
  }
  warnRemovedEntryFact(EntryFact);
  if (Ctx.ExitLEK == LEK_LockedSomePredecessors)
    EntrySet.removeFact(FactMan, EntryFact);
}

// Mixed, the exit side's hold meeting only conditional try facts on the entry
// side. With the exit side holding definitely alone: forgiven under
// mixedJoinExempt(), else diagnosed as not held on the other path; the merged
// state is the other side's. With both sides holding conditionally and this one
// definitely as well: the demotion decides, and where none applies the
// difference is read as a reentrancy-depth mismatch, keeping the deeper state
// to minimize follow-on warnings.
void LocksetJoin::joinMixedFromExit(FactID Fact, const FactEntry &ExitFact,
                                    bool ExitHasCond) {
  if (!ExitHasCond) {
    if (!mixedJoinExempt(ExitSet, ExitFact, EntrySetOrig))
      warnRemovedExitFact(ExitFact);
    else if (Ctx.canModify() && reentrancyDepth(ExitFact) != 0)
      warnReentrancyMismatch(ExitFact, Ctx.EntryLEK);
    return;
  }
  // The other side holds the capability conditionally and not definitely,
  // so this side's definite hold is one-sided whatever this side also
  // carries: where it demotes -- the other side records its own call's
  // failure, or the terminator rebranches on that call -- the join is that
  // call's conditional try fact, not a level to keep. Two try-acquires of
  // one capability, each branched on, meet here: the second call's proved
  // hold beside the first call's unresolved try fact.
  if (std::optional<Demotion> D =
          holdDemotion(ExitSet, EntrySetOrig, ExitFact, Ctx.EntryLEK)) {
    if (Ctx.canModify())
      demoteExitFact(ExitFact, D->Origin, Ctx.EntryLEK);
    else
      warnRemovedExitFact(ExitFact);
    return;
  }
  warnReentrancyMismatch(ExitFact, Ctx.EntryLEK);
  if (Ctx.canModify())
    EntrySet.addLockByID(Fact);
}

// Mixed, the entry side's hold meeting only conditional try facts on the exit
// side (kept by joinTryFactFromExit()). With the entry side holding definitely
// alone: diagnosed unless forgiven, and giving way to them. With both sides
// conditional and this one definitely deeper: diagnosed from the exit side's
// view by joinMixedFromExit() too; the deeper state is kept.
void LocksetJoin::joinMixedFromEntry(const FactEntry &EntryFact,
                                     bool EntryHasCond) {
  if (EntryHasCond) {
    // As in joinMixedFromExit(): the demotion decides first, since the
    // other side does not hold the capability definitely at all.
    if (std::optional<Demotion> D =
            holdDemotion(EntrySetOrig, ExitSet, EntryFact, Ctx.ExitLEK)) {
      if (Ctx.canModify())
        demoteEntryFactInPlace(EntryFact, D->Origin, Ctx.ExitLEK);
      else
        warnRemovedEntryFact(EntryFact);
      return;
    }
    warnReentrancyMismatch(EntryFact, Ctx.ExitLEK);
    return;
  }
  const TryFactEntry *Proof = mixedJoinExempt(EntrySetOrig, EntryFact, ExitSet);
  if (!Proof)
    warnRemovedEntryFact(EntryFact);
  else if (Ctx.canModify() && reentrancyDepth(EntryFact) != 0)
    warnReentrancyMismatch(EntryFact, Ctx.ExitLEK);
  if (Ctx.ExitLEK == LEK_LockedSomePredecessors)
    EntrySet.removeFact(FactMan, EntryFact);
  else if (Proof && Ctx.isUnsealedLoopJoin())
    demoteEntryFactInPlace(EntryFact, Proof->origin(), Ctx.ExitLEK);
}

void LocksetJoin::run() {
  // Facts of the exit set: paired with their counterpart in the entry set, and
  // joined or handled one-sided.
  for (FactID Fact : ExitSet) {
    const FactEntry &ExitFact = FactMan[Fact];
    FactSet::iterator EntryIt = EntrySet.findCounterpartIter(FactMan, ExitFact);
    if (const auto *W = dyn_cast<TryFactEntry>(&ExitFact)) {
      if (EntryIt != EntrySet.end())
        joinTryFactPair(EntryIt, *W);
      else
        joinTryFactFromExit(Fact, *W);
    } else if (EntryIt != EntrySet.end()) {
      joinDefinitePair(EntryIt, Fact, ExitFact);
    } else {
      joinDefiniteFromExit(Fact, ExitFact);
    }
  }

  // Facts of the entry set without a counterpart in the exit set.
  for (FactID Fact : EntrySetOrig) {
    const FactEntry &EntryFact = FactMan[Fact];
    if (ExitSet.findCounterpart(FactMan, EntryFact))
      continue;
    if (const auto *W = dyn_cast<TryFactEntry>(&EntryFact))
      joinTryFactFromEntry(*W);
    else
      joinDefiniteFromEntry(EntryFact);
  }
}

/// Compute the intersection of two locksets and issue warnings for any
/// locks in the symmetric difference.
///
/// This function is used at a merge point in the CFG when comparing the lockset
/// of each branch being merged. For example, given the following sequence:
/// A; if () then B; else C; D; we need to check that the lockset after B and C
/// are the same. In the event of a difference, we use the intersection of these
/// two locksets at the start of D.
///
/// \param EntrySet A lockset for entry into a (possibly new) block.
/// \param ExitSet The lockset on exiting a preceding block.
/// \param Ctx Where the join is, which kind of join it is, and what the
/// joining block does with the try-acquire results the sets carry (see
/// LocksetJoin).
void ThreadSafetyAnalyzer::intersectAndWarn(FactSet &EntrySet,
                                            const FactSet &ExitSet,
                                            const JoinContext &Ctx) {
  LocksetJoin(*this, EntrySet, ExitSet, Ctx).run();
}

// Return true if block B never continues to its successors.
static bool neverReturns(const CFGBlock *B) {
  if (B->hasNoReturnElement())
    return true;
  if (B->empty())
    return false;

  CFGElement Last = B->back();
  if (std::optional<CFGStmt> S = Last.getAs<CFGStmt>()) {
    if (isa<CXXThrowExpr>(S->getStmt()))
      return true;
  }

  // If B constructed a temporary whose destructor is noreturn, control entering
  // the decision block will always branch to the non-returning destructor.
  if (B->succ_size() == 1) {
    if (const CFGBlock *Succ = *B->succ_begin()) {
      if (Succ->getTerminator().isTemporaryDtorsBranch() &&
          Succ->succ_size() == 2) {
        // The decision block's terminator is the CXXBindTemporaryExpr; if B
        // bound this temporary, entering Succ from B takes the true (dtor)
        // edge; otherwise it takes the false (alternative dtor / continuation)
        // edge.
        const Stmt *Term = Succ->getTerminatorStmt();
        bool Bound = llvm::any_of(*B, [Term](const CFGElement &CE) {
          auto CS = CE.getAs<CFGStmt>();
          return CS && CS->getStmt() == Term;
        });
        if (const auto *Next =
                (Bound ? *Succ->succ_begin() : *(Succ->succ_begin() + 1))
                    .getReachableBlock())
          return neverReturns(Next);
      }
    }
  }

  return false;
}

/// The same capability listed under opposite success values -- of either
/// lock kind -- is acquired regardless of the call's result: move it out
/// of the polarity groups into an unconditional group, leaving every
/// remaining capability recorded under one polarity only. A capability
/// listed twice under the same polarity keeps both of its kinds; the
/// acquisition itself decides what to do with the second (addTryLock()).
/// Exclusive under both polarities stays exclusive. A cross-kind pairing
/// (e.g. exclusive on success, shared on failure) may be a deliberate
/// API, but a single fact cannot represent a hold whose kind varies with
/// the result, so it keeps only the guarantee that holds either way: an
/// unconditional shared hold. handleCall() adds the unconditional groups
/// to the lockset, with the diagnostic.
/// "Regardless" is a truthiness conclusion, so it holds only when the
/// two polarities cover the result's domain: always for a boolean
/// result, and for an integer result whose truthy side promises any
/// nonzero value. A truthy side keyed to specific integer codes does
/// not cover -- TRY_ACQUIRE(1, mu) TRY_ACQUIRE(0, mu) on an int result
/// acquires mu iff the result is 0 or 1, and a result of 2 acquires
/// nothing -- so the capability stays conditional, recorded under both
/// polarities, and the edges resolve it by value (resolveTrylockEdge()).
void ThreadSafetyAnalyzer::reconcileTryAcquireCaps(const Expr *Exp,
                                                   TryAcquireCaps &Caps) {
  const auto *CE = dyn_cast_if_present<CallExpr>(Exp);
  const bool BoolResult = CE && CE->isKnownToHaveBooleanValue();
  auto CoversResultDomain = [&](const CapabilityExpr &M) {
    // A call with no result to branch on -- a constructor, or an
    // expression-less one (a cleanup function) -- covers its domain
    // vacuously: nothing can ever resolve the conditional fact the
    // codes would keep, and both polarities acquire the capability.
    // Otherwise the truthy side covers it exactly when it promises the
    // acquisition on any nonzero result (CapProfile::AnyNonzero).
    return !CE || BoolResult || getCapProfile(Caps, M).AnyNonzero;
  };
  // A capability promised on both outcomes is acquired regardless of the
  // result, in the weaker kind if the two outcomes disagree about it.
  for (const CapabilityExpr &M : Caps.TruthyExclusive) {
    if (!CoversResultDomain(M))
      continue;
    if (Caps.FalsyExclusive.contains(M))
      Caps.UnconditionalExclusive.push_back_nodup(M);
    else if (Caps.FalsyShared.contains(M))
      Caps.UnconditionalShared.push_back_nodup(M);
  }
  for (const CapabilityExpr &M : Caps.TruthyShared)
    if (!Caps.UnconditionalExclusive.contains(M) &&
        (Caps.FalsyExclusive.contains(M) || Caps.FalsyShared.contains(M)) &&
        CoversResultDomain(M))
      Caps.UnconditionalShared.push_back_nodup(M);
  if (Caps.UnconditionalExclusive.empty() && Caps.UnconditionalShared.empty())
    return;
  auto DropRegardless = [&](CapExprSet &Set) {
    llvm::erase_if(Set, [&](const CapabilityExpr &M) {
      return Caps.UnconditionalExclusive.contains(M) ||
             Caps.UnconditionalShared.contains(M);
    });
  };
  DropRegardless(Caps.TruthyExclusive);
  DropRegardless(Caps.TruthyShared);
  DropRegardless(Caps.FalsyExclusive);
  DropRegardless(Caps.FalsyShared);
}

/// Record the capabilities named by the try-acquire attributes of the call
/// or construction \p Exp to \p D into TryAcquireCapsMap, translated in the
/// currently installed context, and reconcile degenerate annotations. A
/// call without an expression (a destructor or cleanup function) records
/// into \p NoExprCaps instead: there is no result to branch on, but a
/// reconciled unconditional acquisition still applies.
ThreadSafetyAnalyzer::TryAcquireCaps *
ThreadSafetyAnalyzer::recordedTryAcquireCaps(const Expr *Exp) {
  auto It = TryAcquireCapsMap.find(Exp);
  return It == TryAcquireCapsMap.end() ? nullptr : &It->second;
}

/// Mark every recorded try-acquire call that names \p Cp as spent
/// (FactManager::spentTryAcquire()): its stored result no longer determines
/// a level, so no branch on it may re-materialize a hold. The call executing
/// again clears the mark (handleCall()).
void ThreadSafetyAnalyzer::spendTryAcquiresOf(const CapabilityExpr &Cp) {
  for (const auto &[Exp, Caps] : TryAcquireCapsMap)
    if (Caps.TruthyExclusive.contains(Cp) || Caps.TruthyShared.contains(Cp) ||
        Caps.FalsyExclusive.contains(Cp) || Caps.FalsyShared.contains(Cp))
      FactMan.addSpentTryAcquire(Exp);
}

/// Whether two try-acquire calls are twins: their recorded capabilities
/// agree group for group, so a variable merging their results holds "the
/// result of acquiring these capabilities" whichever path assigned it, and
/// a branch on it resolves the same facts either way. Decided on the
/// record and not on the syntax, because two calls that read alike can
/// acquire different capabilities -- `Mutex *p = &m1; ok = p->TryLock();
/// p = &m2; ok = p->TryLock();` profiles identically on both paths and
/// resolving the second against the first's capabilities asserts (in a
/// release build, silently resolves a genuine hold as a failure).
bool ThreadSafetyAnalyzer::sameTryAcquireCaps(const Expr *A, const Expr *B) {
  const TryAcquireCaps *CA = recordedTryAcquireCaps(A);
  const TryAcquireCaps *CB = recordedTryAcquireCaps(B);
  if (!CA || !CB)
    return false;
  auto SameSet = [](const CapExprSet &X, const CapExprSet &Y) {
    return X.size() == Y.size() &&
           llvm::all_of(X,
                        [&](const CapabilityExpr &C) { return Y.contains(C); });
  };
  return SameSet(CA->TruthyExclusive, CB->TruthyExclusive) &&
         SameSet(CA->TruthyShared, CB->TruthyShared) &&
         SameSet(CA->FalsyExclusive, CB->FalsyExclusive) &&
         SameSet(CA->FalsyShared, CB->FalsyShared) &&
         SameSet(CA->UnconditionalExclusive, CB->UnconditionalExclusive) &&
         SameSet(CA->UnconditionalShared, CB->UnconditionalShared);
}

ThreadSafetyAnalyzer::TryAcquireCaps &
ThreadSafetyAnalyzer::recordTryAcquireCall(const Expr *Exp, const NamedDecl *D,
                                           til::SExpr *Self,
                                           TryAcquireCaps *NoExprCaps) {
  assert((Exp || NoExprCaps) && "expression-less call without a caps store");
  TryAcquireCaps &Caps = Exp ? TryAcquireCapsMap[Exp] : *NoExprCaps;
  ASTContext &Ctx = D->getASTContext();
  const auto *FD = dyn_cast<FunctionDecl>(D);
  const QualType ResultTy = FD ? FD->getReturnType() : QualType();
  // Each attribute's success value, decoded once: the polarity it reports
  // acquisition on, and the exact result code it names, if any.
  struct AttrSuccess {
    const TryAcquireCapabilityAttr *A;
    bool Truthy;
    std::optional<llvm::APSInt> Code;
  };
  SmallVector<AttrSuccess, 2> Attrs;
  for (const Attr *At : D->attrs()) {
    const auto *A = dyn_cast<TryAcquireCapabilityAttr>(At);
    if (!A)
      continue;
    const bool Truthy = getTrySuccessValue(Ctx, A->getSuccessValue());
    Attrs.push_back(
        {A, Truthy,
         Truthy ? getTrySuccessCode(Ctx, A->getSuccessValue(), ResultTy)
                : std::nullopt});
  }
  // A truthy success value keys the acquisition to that exact result value
  // only where the call discriminates its outcomes by value at all: some
  // attribute names a value other than 1, or two of them name distinct
  // ones. Success reported as a plain 1 says no more than "nonzero" --
  // before C23 <stdbool.h> spells `true` that way, and the Linux kernel's
  // `__cond_acquires(nonzero, x)` expands to it -- so a call naming no
  // other value keys its capabilities to truthiness, as every try-acquire
  // did before codes existed, and only a call that names another value
  // means "this nonzero result and not that one". Where codes are kept,
  // the decode folds `== code` comparisons, and the edges resolve case
  // labels and label exclusions, against them.
  const bool KeyByCode = llvm::any_of(
      Attrs, [](const AttrSuccess &S) { return S.Code && *S.Code != 1; });
  for (const auto &[A, Truthy, AttrCode] : Attrs) {
    CapExprSet &Group =
        Truthy ? (A->isShared() ? Caps.TruthyShared : Caps.TruthyExclusive)
               : (A->isShared() ? Caps.FalsyShared : Caps.FalsyExclusive);
    CapExprSet AttrCaps;
    getMutexIDs(AttrCaps, A, Exp, D, Self);
    const std::optional<llvm::APSInt> &Code =
        KeyByCode ? AttrCode : std::nullopt;
    for (const auto &M : AttrCaps) {
      Group.push_back_nodup(M);
      if (!Truthy)
        continue;
      if (!Code)
        Caps.TruthyAny.push_back_nodup(M);
      else if (llvm::none_of(Caps.ExactCodes, [&](const auto &Recorded) {
                 return llvm::APSInt::isSameValue(Recorded.second, *Code) &&
                        Recorded.first.equals(M);
               }))
        Caps.ExactCodes.emplace_back(M, *Code);
    }
  }
  reconcileTryAcquireCaps(Exp, Caps);
  // Only the conditional groups: a capability reconcile moved to the
  // unconditional ones is acquired outright, so a negative fact of it at
  // the call is a real release and keeps its note.
  if (Exp)
    for (const CapExprSet *Group : {&Caps.TruthyExclusive, &Caps.TruthyShared,
                                    &Caps.FalsyExclusive, &Caps.FalsyShared})
      for (const CapabilityExpr &M : *Group)
        FactMan.addTryAcquireLoc(Exp->getExprLoc(), M);
  return Caps;
}

/// Populate TryAcquireCapsMap for every try-acquire CallExpr in the
/// function, before the lockset walk: a branch on a stored result can
/// precede the call in block order (a loop-top check `if (ok)` above
/// `ok = mu.TryLock()`), and the terminator decode (decodeTrylockBranch)
/// folds the record into its memoized per-capability resolutions. The
/// variable map listed the calls with their post-contexts as it was
/// built, so each call's attributes translate in the call's own context.
/// Constructors are excluded: they record in handleCall, where the
/// constructed-object placeholder is available.
void ThreadSafetyAnalyzer::recordTryAcquireCalls() {
  const bool Beta = Handler.issueBetaWarnings();
  llvm::scope_exit Cleanup([&] {
    if (Beta)
      SxBuilder.setLookupLocalVarExpr(nullptr);
  });
  for (const auto &[CE, Ctx] : LocalVarMap.tryAcquireCalls()) {
    // Mirror BuildLockset's post-context attribute translation.
    if (Beta)
      SxBuilder.setLookupLocalVarExpr(
          [Ctx = Ctx, this](const NamedDecl *VD) mutable -> const Expr * {
            return LocalVarMap.lookupExpr(VD, Ctx);
          });
    recordTryAcquireCall(CE, cast<NamedDecl>(CE->getCalleeDecl()));
  }
}

/// Record the release evidence reaching a loop's latch -- a Released try
/// fact, or a conditional one marked MayBeReleased by an inner loop's back
/// edge or a branch join inside the body -- in the sealed exit sets its
/// exit edges are computed from: an iteration may have released the
/// capability, and the blocks were analyzed before this back edge was
/// seen, so the exit edges would otherwise resolve or re-materialize a
/// hold the loop may have released (getEdgeLockset()).
void ThreadSafetyAnalyzer::injectLoopReleasedTryFacts(
    const CFGBlock *Head, const CFGBlock *Latch,
    PostOrderCFGView::CFGBlockSet &Visited) {
  if (TryAcquireCapsMap.empty())
    return;
  SmallVector<FactID, 2> Injectable;
  for (const auto &Fact : BlockInfo[Latch->getBlockID()].ExitSet) {
    const FactEntry &FE = FactMan[Fact];
    if (const auto *W = dyn_cast<TryFactEntry>(&FE);
        W && (W->released() || (W->conditional() && W->mayBeReleased())))
      Injectable.push_back(Fact);
  }
  if (Injectable.empty())
    return;

  // The loop's blocks: those reaching the latch backwards without passing
  // the head, bounded by what the head reaches forwards. Without that
  // bound a second entry into the body (a goto into the loop, or an
  // irreducible loop) would let the backward walk escape the loop and
  // absorb arbitrary blocks up to the function's entry.
  llvm::SmallPtrSet<const CFGBlock *, 16> FromHead;
  SmallVector<const CFGBlock *, 16> Work{Head};
  FromHead.insert(Head);
  while (!Work.empty()) {
    const CFGBlock *B = Work.pop_back_val();
    for (CFGBlock::const_succ_iterator SI = B->succ_begin(), SE = B->succ_end();
         SI != SE; ++SI)
      if (*SI && FromHead.insert(*SI).second)
        Work.push_back(*SI);
  }
  llvm::SmallPtrSet<const CFGBlock *, 8> LoopBlocks;
  LoopBlocks.insert(Head);
  LoopBlocks.insert(Latch);
  Work.assign({Latch});
  while (!Work.empty()) {
    const CFGBlock *B = Work.pop_back_val();
    for (CFGBlock::const_pred_iterator PI = B->pred_begin(), PE = B->pred_end();
         PI != PE; ++PI)
      if (*PI && FromHead.contains(*PI) && LoopBlocks.insert(*PI).second)
        Work.push_back(*PI);
  }

  // Any member owning an exit edge needs the evidence (the head, a break,
  // a goto out), and so does a sealed block outside the loop that an exit
  // edge passes through: a break statement's own block does not reach the
  // latch, yet its exit set feeds the post-loop join. An unvisited
  // successor needs nothing -- its entry derives from a patched set later.
  SmallVector<const CFGBlock *, 8> PatchBlocks;
  llvm::SmallPtrSet<const CFGBlock *, 8> SealedOutside;
  SmallVector<const CFGBlock *, 4> OutsideWork;
  auto NoteOutsideSucc = [&](const CFGBlock *S) {
    if (S && !LoopBlocks.contains(S) && Visited.alreadySet(S) &&
        SealedOutside.insert(S).second)
      OutsideWork.push_back(S);
  };
  for (const CFGBlock *B : LoopBlocks) {
    bool HasExitEdge = false;
    for (CFGBlock::const_succ_iterator SI = B->succ_begin(), SE = B->succ_end();
         SI != SE; ++SI) {
      if (*SI && !LoopBlocks.contains(*SI))
        HasExitEdge = true;
      NoteOutsideSucc(*SI);
    }
    if (HasExitEdge)
      PatchBlocks.push_back(B);
  }
  while (!OutsideWork.empty()) {
    const CFGBlock *B = OutsideWork.pop_back_val();
    PatchBlocks.push_back(B);
    for (CFGBlock::const_succ_iterator SI = B->succ_begin(), SE = B->succ_end();
         SI != SE; ++SI)
      NoteOutsideSucc(*SI);
  }

  for (const CFGBlock *B : PatchBlocks) {
    FactSet &Target = BlockInfo[B->getBlockID()].ExitSet;
    for (FactID Fact : Injectable) {
      const auto *W = cast<TryFactEntry>(&FactMan[Fact]);
      // A conditional try fact of the same call in the target is marked
      // MayBeReleased with the back edge's release: the loop's exit edges must
      // not resolve a result the body may have released (getEdgeLockset()),
      // while the possible hold of the iteration that did not release it is
      // still diagnosed. A resolved try fact stays as it is -- what the head's
      // own iteration proved or failed is not refuted by a later iteration's
      // release. A target without any try fact of the call takes a Released
      // one (resolved: it adds no possible hold), but not a conditional one,
      // which would.
      if (FactSet::iterator It =
              Target.findTryFactIter(FactMan, *W, W->origin(), W->kind());
          It != Target.end()) {
        const auto &Existing = cast<TryFactEntry>(FactMan[*It]);
        if (Existing.conditional() && !Existing.mayBeReleased())
          Target.replaceFact(
              FactMan, It, Existing.asMayBeReleased(FactMan, W->releaseLoc()));
        continue;
      }
      if (W->released())
        Target.addLockByID(Fact);
    }
  }
}

/// Check a function's CFG for thread-safety violations.
///
/// We traverse the blocks in the CFG, compute the set of mutexes that are held
/// at the end of each block, and issue warnings for thread safety violations.
/// Each block in the CFG is traversed exactly once.
void ThreadSafetyAnalyzer::runAnalysis(AnalysisDeclContext &AC) {
  // TODO: this whole function needs be rewritten as a visitor for CFGWalker.
  // For now, we just use the walker to set things up.
  threadSafety::CFGWalker walker;
  if (!walker.init(AC))
    return;

  // AC.dumpCFG(true);
  // threadSafety::printSCFG(walker);

  CFG *CFGraph = walker.getGraph();
  const NamedDecl *D = walker.getDecl();
  CurrentFunction = dyn_cast<FunctionDecl>(D);
  ASTCtx = &D->getASTContext();

  if (D->hasAttr<NoThreadSafetyAnalysisAttr>())
    return;

  // FIXME: Do something a bit more intelligent inside constructor and
  // destructor code.  Constructors and destructors must assume unique access
  // to 'this', so checks on member variable access is disabled, but we should
  // still enable checks on other objects.
  if (isa<CXXConstructorDecl>(D))
    return;  // Don't check inside constructors.
  if (isa<CXXDestructorDecl>(D))
    return;  // Don't check inside destructors.

  Handler.enterFunction(CurrentFunction);

  BlockInfo.resize(CFGraph->getNumBlockIDs(),
    CFGBlockInfo::getEmptyBlockInfo(LocalVarMap));

  // We need to explore the CFG via a "topological" ordering.
  // That way, we will be guaranteed to have information about required
  // predecessor locksets when exploring a new block.
  const PostOrderCFGView *SortedGraph = walker.getSortedGraph();
  PostOrderCFGView::CFGBlockSet VisitedBlocks(CFGraph);

  CFGBlockInfo &Initial = BlockInfo[CFGraph->getEntry().getBlockID()];
  CFGBlockInfo &Final   = BlockInfo[CFGraph->getExit().getBlockID()];

  // Mark entry block as reachable
  Initial.Reachable = true;

  // Compute SSA names for local variables
  LocalVarMap.traverseCFG(AC, CFGraph, SortedGraph, BlockInfo,
                          Handler.issueBetaWarnings());

  // Fill in source locations for all CFGBlocks.
  findBlockLocations(CFGraph, SortedGraph, BlockInfo);

  CapExprSet ExclusiveLocksAcquired;
  CapExprSet SharedLocksAcquired;
  CapExprSet LocksReleased;

  // Add locks from exclusive_locks_required and shared_locks_required
  // to initial lockset. Also turn off checking for lock and unlock functions.
  // FIXME: is there a more intelligent way to check lock/unlock functions?
  if (!SortedGraph->empty()) {
    assert(*SortedGraph->begin() == &CFGraph->getEntry());
    FactSet &InitialLockset = Initial.EntrySet;

    CapExprSet ExclusiveLocksToAdd;
    CapExprSet SharedLocksToAdd;

    SourceLocation Loc = D->getLocation();
    for (const auto *Attr : D->attrs()) {
      Loc = Attr->getLocation();
      if (const auto *A = dyn_cast<RequiresCapabilityAttr>(Attr)) {
        getMutexIDs(A->isShared() ? SharedLocksToAdd : ExclusiveLocksToAdd, A,
                    nullptr, D);
      } else if (const auto *A = dyn_cast<ReleaseCapabilityAttr>(Attr)) {
        // UNLOCK_FUNCTION() is used to hide the underlying lock implementation.
        // We must ignore such methods.
        if (A->args_size() == 0)
          return;
        getMutexIDs(A->isShared() ? SharedLocksToAdd : ExclusiveLocksToAdd, A,
                    nullptr, D);
        getMutexIDs(LocksReleased, A, nullptr, D);
      } else if (const auto *A = dyn_cast<AcquireCapabilityAttr>(Attr)) {
        if (A->args_size() == 0)
          return;
        getMutexIDs(A->isShared() ? SharedLocksAcquired
                                  : ExclusiveLocksAcquired,
                    A, nullptr, D);
      } else if (isa<TryAcquireCapabilityAttr>(Attr)) {
        // Don't try to check trylock functions for now.
        return;
      }
    }
    ArrayRef<ParmVarDecl *> Params;
    if (CurrentFunction)
      Params = CurrentFunction->getCanonicalDecl()->parameters();
    else if (auto CurrentMethod = dyn_cast<ObjCMethodDecl>(D))
      Params = CurrentMethod->getCanonicalDecl()->parameters();
    else
      llvm_unreachable("Unknown function kind");
    for (const ParmVarDecl *Param : Params) {
      if (isCallbackParam(Param))
        continue;
      CapExprSet UnderlyingLocks;
      for (const auto *Attr : Param->attrs()) {
        Loc = Attr->getLocation();
        if (const auto *A = dyn_cast<ReleaseCapabilityAttr>(Attr)) {
          getMutexIDs(A->isShared() ? SharedLocksToAdd : ExclusiveLocksToAdd, A,
                      nullptr, Param);
          getMutexIDs(LocksReleased, A, nullptr, Param);
          getMutexIDs(UnderlyingLocks, A, nullptr, Param);
        } else if (const auto *A = dyn_cast<RequiresCapabilityAttr>(Attr)) {
          getMutexIDs(A->isShared() ? SharedLocksToAdd : ExclusiveLocksToAdd, A,
                      nullptr, Param);
          getMutexIDs(UnderlyingLocks, A, nullptr, Param);
        } else if (const auto *A = dyn_cast<AcquireCapabilityAttr>(Attr)) {
          getMutexIDs(A->isShared() ? SharedLocksAcquired
                                    : ExclusiveLocksAcquired,
                      A, nullptr, Param);
          getMutexIDs(UnderlyingLocks, A, nullptr, Param);
        } else if (const auto *A = dyn_cast<LocksExcludedAttr>(Attr)) {
          getMutexIDs(UnderlyingLocks, A, nullptr, Param);
        }
      }
      if (UnderlyingLocks.empty())
        continue;
      CapabilityExpr Cp(SxBuilder.translateVariable(Param, nullptr),
                        StringRef(),
                        /*Neg=*/false, /*Reentrant=*/false);
      auto *ScopedEntry = FactMan.createFact<ScopedLockableFactEntry>(
          Cp, Param->getLocation(), FactEntry::Declared,
          UnderlyingLocks.size());
      for (const CapabilityExpr &M : UnderlyingLocks)
        ScopedEntry->addLock(M);
      addLock(InitialLockset, ScopedEntry, true);
    }

    // FIXME -- Loc can be wrong here.
    for (const auto &Mu : ExclusiveLocksToAdd) {
      const auto *Entry = FactMan.createFact<LockableFactEntry>(
          Mu, LK_Exclusive, Loc, FactEntry::Declared);
      addLock(InitialLockset, Entry, true);
    }
    for (const auto &Mu : SharedLocksToAdd) {
      const auto *Entry = FactMan.createFact<LockableFactEntry>(
          Mu, LK_Shared, Loc, FactEntry::Declared);
      addLock(InitialLockset, Entry, true);
    }
  }

  // Record the capabilities of every try-acquire call, recorded in the exact
  // context of that call.
  recordTryAcquireCalls();

  // Compute the expected exit set.
  // By default, we expect all locks held on entry to be held on exit.
  FactSet ExpectedFunctionExitSet = Initial.EntrySet;

  // Adjust the expected exit set by adding or removing locks, as declared
  // by *-LOCK_FUNCTION and UNLOCK_FUNCTION.  The intersect below will then
  // issue the appropriate warning.
  // FIXME: the location here is not quite right.
  for (const auto &Lock : ExclusiveLocksAcquired)
    ExpectedFunctionExitSet.addLock(
        FactMan, FactMan.createFact<LockableFactEntry>(Lock, LK_Exclusive,
                                                       D->getLocation()));
  for (const auto &Lock : SharedLocksAcquired)
    ExpectedFunctionExitSet.addLock(
        FactMan, FactMan.createFact<LockableFactEntry>(Lock, LK_Shared,
                                                       D->getLocation()));
  for (const auto &Lock : LocksReleased)
    ExpectedFunctionExitSet.removeDefinite(FactMan, Lock);

  for (const auto *CurrBlock : *SortedGraph) {
    unsigned CurrBlockID = CurrBlock->getBlockID();
    CFGBlockInfo *CurrBlockInfo = &BlockInfo[CurrBlockID];

    // Use the default initial lockset in case there are no predecessors.
    VisitedBlocks.insert(CurrBlock);

    // Iterate through the predecessor blocks and warn if the lockset for all
    // predecessors is not the same. We take the entry lockset of the current
    // block to be the intersection of all previous locksets.
    // FIXME: By keeping the intersection, we may output more errors in future
    // for a lock which is not in the intersection, but was in the union. We
    // may want to also keep the union in future. As an example, let's say
    // the intersection contains Mutex L, and the union contains L and M.
    // Later we unlock M. At this point, we would output an error because we
    // never locked M; although the real error is probably that we forgot to
    // lock M on all code paths. Conversely, let's say that later we lock M.
    // In this case, we should compare against the intersection instead of the
    // union because the real error is probably that we forgot to unlock M on
    // all code paths.
    bool LocksetInitialized = false;
    // The branch-join context. Its try-acquire call -- the one whose
    // result the condition starting at this block branches on, if any --
    // is computed lazily on the first join where a set carries a try fact
    // at all. Each incoming set is scanned once as it arrives
    // (JoinHasTryLockFact accumulates); the entry set itself never gains
    // try facts from anywhere else.
    JoinContext Ctx{CurrBlockInfo->EntryLoc, LEK_LockedSomePredecessors,
                    LEK_LockedSomePredecessors};
    bool RebranchTryLockComputed = false;
    bool JoinHasTryLockFact = false;
    auto HasTryLockFact = [this](const FactSet &FS) {
      // Functions without a try-acquire (the common case) record none:
      // skip scanning the fact sets entirely. (A construction records
      // in-walk, but before any of its try facts can reach a set.)
      return !TryAcquireCapsMap.empty() && llvm::any_of(FS, [this](FactID ID) {
        return isa<TryFactEntry>(FactMan[ID]);
      });
    };
    // The lockset of the first infeasible incoming edge, if any (see below).
    std::optional<FactSet> InfeasibleEdgeSet;
    for (CFGBlock::const_pred_iterator PI = CurrBlock->pred_begin(),
         PE  = CurrBlock->pred_end(); PI != PE; ++PI) {
      // if *PI -> CurrBlock is a back edge
      if (*PI == nullptr || !VisitedBlocks.alreadySet(*PI))
        continue;

      unsigned PrevBlockID = (*PI)->getBlockID();
      CFGBlockInfo *PrevBlockInfo = &BlockInfo[PrevBlockID];

      // Ignore edges from blocks that can't return.
      if (neverReturns(*PI) || !PrevBlockInfo->Reachable)
        continue;

      FactSet PrevLockset;
      if (getEdgeLockset(PrevLockset, PrevBlockInfo->ExitSet, *PI, CurrBlock) ||
          PrevBlockInfo->CoverageOnly) {
        // The edge cannot be taken (a resolved try fact contradicts it), or
        // the predecessor itself was analyzed only for coverage and its exit
        // set is dead state either way: skip
        // the edge at the join like an unreachable predecessor. Remember
        // the lockset in case no live predecessor remains: infeasibility
        // only prunes joins, never analysis coverage (see below).
        if (!InfeasibleEdgeSet)
          InfeasibleEdgeSet = std::move(PrevLockset);
        continue;
      }

      // Okay, we can reach this block from the entry.
      CurrBlockInfo->Reachable = true;

      if (!LocksetInitialized) {
        CurrBlockInfo->EntrySet = PrevLockset;
        JoinHasTryLockFact = HasTryLockFact(PrevLockset);
        LocksetInitialized = true;
      } else {
        // Surprisingly 'continue' doesn't always produce back edges, because
        // the CFG has empty "transition" blocks where they meet with the end
        // of the regular loop body. We still want to diagnose them as loop.
        if (isa_and_nonnull<ContinueStmt>((*PI)->getTerminatorStmt())) {
          // Loop join: warn on locks held for only some iterations.
          intersectAndWarn(CurrBlockInfo->EntrySet, PrevLockset,
                           CurrBlockInfo->EntryLoc,
                           LEK_LockedSomeLoopIterations);
        } else {
          // Branch join: a difference in the holds a try-acquire's try facts
          // prove is demoted to conditional and re-resolved on the outgoing
          // edges if the condition branches on that call's result --
          // possibly behind short-circuit blocks of a compound condition
          // like `c && ok`.
          if (!RebranchTryLockComputed && !JoinHasTryLockFact)
            JoinHasTryLockFact = HasTryLockFact(PrevLockset);
          if (!RebranchTryLockComputed && JoinHasTryLockFact) {
            // Compute once; the result depends only on CurrBlock, not on *PI.
            // Skipped entirely (the common case) until some try fact reaches
            // this join.
            Ctx.setRebranch(
                getConditionTrylockCallExpr(CurrBlock, /*CheckAllPaths=*/true));
            RebranchTryLockComputed = true;
          }
          intersectAndWarn(CurrBlockInfo->EntrySet, PrevLockset, Ctx);
        }
      }
    }

    // A block reached only through infeasible edges is dynamically dead if
    // the infeasibility proofs are right -- but the proof rests on the
    // local-variable map, which can be stale (e.g. a result variable
    // mutated through an escaped reference), and even genuinely dead code
    // gets its diagnostics. So analyze the block anyway, with one of the
    // infeasible edges' locksets: infeasibility prunes joins, never
    // analysis coverage. The block is marked coverage-only, which
    // quarantines its exit set from downstream joins (above) and
    // propagates through blocks reachable only from it.
    if (!CurrBlockInfo->Reachable && InfeasibleEdgeSet) {
      CurrBlockInfo->Reachable = true;
      CurrBlockInfo->CoverageOnly = true;
      CurrBlockInfo->EntrySet = std::move(*InfeasibleEdgeSet);
    }

    // Skip rest of block if it's not reachable.
    if (!CurrBlockInfo->Reachable)
      continue;

    BuildLockset LocksetBuilder(this, *CurrBlockInfo, ExpectedFunctionExitSet);

    // Visit all the statements in the basic block.
    for (const auto &BI : *CurrBlock) {
      switch (BI.getKind()) {
        case CFGElement::Statement: {
          CFGStmt CS = BI.castAs<CFGStmt>();
          LocksetBuilder.Visit(CS.getStmt());
          break;
        }
        // Ignore BaseDtor and MemberDtor for now.
        case CFGElement::AutomaticObjectDtor: {
          CFGAutomaticObjDtor AD = BI.castAs<CFGAutomaticObjDtor>();
          const auto *DD = AD.getDestructorDecl(AC.getASTContext());
          // Function parameters as they are constructed in caller's context and
          // the CFG does not contain the ctors. Ignore them as their
          // capabilities cannot be analysed because of this missing
          // information.
          if (isa_and_nonnull<ParmVarDecl>(AD.getVarDecl()))
            break;
          if (!DD || !DD->hasAttrs())
            break;

          LocksetBuilder.handleCall(
              nullptr, DD,
              SxBuilder.translateVariable(AD.getVarDecl(), nullptr),
              AD.getTriggerStmt()->getEndLoc());
          break;
        }

        case CFGElement::CleanupFunction: {
          const CFGCleanupFunction &CF = BI.castAs<CFGCleanupFunction>();
          LocksetBuilder.handleCall(
              /*Exp=*/nullptr, CF.getFunctionDecl(),
              SxBuilder.translateVariable(CF.getVarDecl(), nullptr),
              CF.getVarDecl()->getLocation());
          break;
        }

        case CFGElement::TemporaryDtor: {
          auto TD = BI.castAs<CFGTemporaryDtor>();

          // Clean up constructed object even if there are no attributes to
          // keep the number of objects in limbo as small as possible.
          if (auto Object = ConstructedObjects.find(
                  TD.getBindTemporaryExpr()->getSubExpr());
              Object != ConstructedObjects.end()) {
            const auto *DD = TD.getDestructorDecl(AC.getASTContext());
            if (DD->hasAttrs())
              // TODO: the location here isn't quite correct.
              LocksetBuilder.handleCall(nullptr, DD, Object->second,
                                        TD.getBindTemporaryExpr()->getEndLoc());
            ConstructedObjects.erase(Object);
          }
          break;
        }
        default:
          break;
      }
    }
    CurrBlockInfo->ExitSet = LocksetBuilder.FSet;

    // A block analyzed only for coverage stops here: its exit set is
    // provably dead state, so back-edge comparisons must not consume it
    // either (the predecessor loop above keeps it out of forward joins).
    if (CurrBlockInfo->CoverageOnly)
      continue;

    // For every back edge from CurrBlock (the end of the loop) to another block
    // (FirstLoopBlock) we need to check that the Lockset of Block is equal to
    // the one held at the beginning of FirstLoopBlock. We can look up the
    // Lockset held at the beginning of FirstLoopBlock in the EntryLockSets map.
    for (CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin(),
         SE  = CurrBlock->succ_end(); SI != SE; ++SI) {
      // if CurrBlock -> *SI is *not* a back edge
      if (*SI == nullptr || !VisitedBlocks.alreadySet(*SI))
        continue;

      CFGBlock *FirstLoopBlock = *SI;
      CFGBlockInfo *PreLoop = &BlockInfo[FirstLoopBlock->getBlockID()];
      CFGBlockInfo *LoopEnd = &BlockInfo[CurrBlockID];
      // A back-edge difference in the holds a try-acquire's try facts prove
      // is forgiven when the loop condition branches on that call's result
      // (e.g. a spin loop storing the result), possibly behind
      // short-circuit blocks of a compound condition: the entry set keeps
      // the (weaker) pre-loop facts and the condition's outgoing edges
      // re-resolve the try fact each iteration, so it does not leak around
      // the loop -- even behind a short-circuit.
      JoinContext Ctx{PreLoop->EntryLoc, LEK_LockedSomeLoopIterations,
                      LEK_LockedSomeLoopIterations};
      Ctx.setRebranch(
          getConditionTrylockCallExpr(FirstLoopBlock, /*CheckAllPaths=*/true));
      Ctx.SealedEntry = true;
      // For the unchecked-result warning: the try-acquire results branched
      // on inside this back edge's natural loop are (or will be, on the
      // next iteration) checked around the loop. Results checked only
      // outside the loop are not: the loop re-executes the call (or
      // discards the result) unchecked.
      llvm::SmallPtrSet<const Expr *, 4> CheckedInLoop;
      if (Handler.issueBetaWarnings() && HasTryLockFact(LoopEnd->ExitSet)) {
        // The natural loop of this back edge: the head, plus every block
        // reaching this latch without passing through the head. (All these
        // blocks precede the latch in the traversal, so their exit contexts
        // are available for the decode below; on an irreducible CFG the
        // walk may escape the loop, erring toward suppression.)
        llvm::SmallPtrSet<const CFGBlock *, 8> LoopBlocks;
        SmallVector<const CFGBlock *, 8> Worklist;
        LoopBlocks.insert(FirstLoopBlock);
        if (LoopBlocks.insert(CurrBlock).second)
          Worklist.push_back(CurrBlock);
        while (!Worklist.empty()) {
          const CFGBlock *B = Worklist.pop_back_val();
          for (CFGBlock::const_pred_iterator BPI = B->pred_begin(),
                                             BPE = B->pred_end();
               BPI != BPE; ++BPI)
            if (*BPI && LoopBlocks.insert(*BPI).second)
              Worklist.push_back(*BPI);
        }
        // Decode each loop block's terminator now, rather than consulting
        // what happened to be decoded already: a goto-rotated loop's latch
        // terminator has not had its forward edges processed yet, and its
        // check must still count. (The decode is memoized, so blocks whose
        // edges were already processed cost a cache hit.)
        for (const CFGBlock *B : LoopBlocks) {
          // A non-void `?:` is a branch on the result, but not a check of
          // it: its arms rejoin at once, and what the merged value says
          // about the result is for a later branch to read -- one this
          // walk counts on its own if the loop holds it. Counting the
          // merge itself calls `mu.TryLock() ? 1 : 2;`, whose value is
          // discarded, a check, and drops the report of the fact its own
          // arm join loses.
          if (const auto *COp =
                  dyn_cast_if_present<AbstractConditionalOperator>(
                      B->getTerminatorStmt());
              COp && !COp->getType()->isVoidType())
            continue;
          TerminatorTrylockCall Checked = getTerminatorTrylockCall(B);
          if (Checked.TrylockCall)
            CheckedInLoop.insert(Checked.TrylockCall);
          // A branch on a merge of two identical calls checks both results.
          if (Checked.MergedCall)
            CheckedInLoop.insert(Checked.MergedCall);
        }
        Ctx.CheckedAroundLoop = &CheckedInLoop;
      }
      intersectAndWarn(PreLoop->EntrySet, LoopEnd->ExitSet, Ctx);
      // A released try fact or negative fact reaching the loop head on its
      // back edge is evidence that an iteration may have released the
      // capability (or failed to re-acquire it): patch it into the sealed
      // exit sets the loop's exit edges are computed from.
      injectLoopReleasedTryFacts(FirstLoopBlock, CurrBlock, VisitedBlocks);
    }
  }

  // Skip the final check only if the exit block is unreachable. A block
  // reachable only through infeasible edges still has to be checked against
  // the function's contract: the check is coverage, and infeasibility prunes
  // what a join consumes, never what is diagnosed -- otherwise a capability
  // with nothing to do with any try-acquire would leak unreported whenever
  // the exit happened to sit behind such a block.
  if (!Final.Reachable)
    return;

  // FIXME: Should we call this function for all blocks which exit the function?
  intersectAndWarn(ExpectedFunctionExitSet, Final.ExitSet,
                   JoinContext{Final.ExitLoc, LEK_LockedAtEndOfFunction,
                               LEK_NotLockedAtEndOfFunction});

  Handler.leaveFunction(CurrentFunction);
}

/// Check a function's CFG for thread-safety violations.
///
/// We traverse the blocks in the CFG, compute the set of mutexes that are held
/// at the end of each block, and issue warnings for thread safety violations.
/// Each block in the CFG is traversed exactly once.
void threadSafety::runThreadSafetyAnalysis(AnalysisDeclContext &AC,
                                           ThreadSafetyHandler &Handler,
                                           BeforeSet **BSet) {
  if (!*BSet)
    *BSet = new BeforeSet;
  ThreadSafetyAnalyzer Analyzer(Handler, *BSet);
  Analyzer.runAnalysis(AC);
}

void threadSafety::threadSafetyCleanup(BeforeSet *Cache) { delete Cache; }

/// Helper function that returns a LockKind required for the given level
/// of access.
LockKind threadSafety::getLockKindFromAccessKind(AccessKind AK) {
  switch (AK) {
    case AK_Read :
      return LK_Shared;
    case AK_Written :
      return LK_Exclusive;
  }
  llvm_unreachable("Unknown AccessKind");
}
