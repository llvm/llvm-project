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
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
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
        St(State::Conditional), MayBeReleased(false) {
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
  bool mayBeReleased() const { return MayBeReleased; }
  SourceLocation releaseLoc() const { return ReleaseLoc; }

  /// This try fact in state \p S, without marks: a Conditional try fact's
  /// MayBeReleased mark is kept only while the state stays. The fact itself
  /// if that is already its state.
  const TryFactEntry *withState(FactManager &FactMan, State S) const;
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

private:
  llvm::SmallDenseMap<SourceLocation, CapExprSet, 4> TryAcquireLocs;
  llvm::SmallDenseSet<const Expr *, 4> SpentTryAcquires;
};

inline const TryFactEntry *TryFactEntry::withState(FactManager &FactMan,
                                                   State S) const {
  if (S == St && !MayBeReleased)
    return this;
  auto *NewFact = FactMan.createFact<TryFactEntry>(*this);
  NewFact->St = S;
  NewFact->MayBeReleased = false;
  return NewFact;
}

const TryFactEntry *TryFactEntry::asReleased(FactManager &FactMan,
                                             SourceLocation Loc) const {
  auto *NewFact = FactMan.createFact<TryFactEntry>(*this);
  NewFact->St = State::Released;
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

/// What a terminator's branch proves about the capabilities of the try-acquire
/// call it branches on.
struct TrylockBranch {
  /// The try-acquire call whose result the terminator branches on, or null if
  /// it does not branch on one.
  const CallExpr *TrylockCall = nullptr;
  /// The call's capabilities for each branch direction. Every capability
  /// recorded for the call appears in both lists, since a direction that does
  /// not prove a capability acquired proves it not acquired.
  SmallVector<TrylockEdgeCap, 1> OnTrue, OnFalse;
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

    // Direct reference to another VarDefinition
    unsigned DirectRef = 0;

    // Reference to underlying canonical non-reference VarDefinition.
    unsigned CanonicalRef = 0;

    // The map with which Exp should be interpreted.
    Context Ctx;

    bool isReference() const { return !Exp; }

    void invalidateRef() { DirectRef = CanonicalRef = 0; }

  private:
    // Create ordinary variable definition
    VarDefinition(const NamedDecl *D, const Expr *E, Context C)
        : Dec(D), Exp(E), Ctx(C) {}

    // Create reference to previous definition
    VarDefinition(const NamedDecl *D, unsigned DirectRef, unsigned CanonicalRef,
                  Context C)
        : Dec(D), DirectRef(DirectRef), CanonicalRef(CanonicalRef), Ctx(C) {}
  };

private:
  Context::Factory ContextFactory;
  std::vector<VarDefinition> VarDefinitions;
  std::vector<std::pair<const Stmt *, Context>> SavedContexts;
  // The function's try-acquire calls, each with the context after it, memoized
  // in traversal order. Constructions are not listed: they record in-walk,
  // where the constructed object's placeholder exists.
  SmallVector<std::pair<const CallExpr *, Context>, 4> TryAcquireCalls;
  // The calls whose result some definition stores in a local variable
  // (storesResultOf()), recorded as the definitions are added.
  llvm::SmallPtrSet<const CallExpr *, 4> StoredResults;

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

  /// Look up the definition for D within the given context.  Returns
  /// NULL if the expression is not statically known.  If successful, also
  /// modifies Ctx to hold the context of the return Expr.
  const Expr* lookupExpr(const NamedDecl *D, Context &Ctx) {
    const unsigned *P = Ctx.lookup(D);
    if (!P)
      return nullptr;

    unsigned i = *P;
    while (i > 0) {
      if (VarDefinitions[i].Exp) {
        Ctx = VarDefinitions[i].Ctx;
        return VarDefinitions[i].Exp;
      }
      i = VarDefinitions[i].DirectRef;
    }
    return nullptr;
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
  void traverseCFG(CFG *CFGraph, const PostOrderCFGView *SortedGraph,
                   std::vector<CFGBlockInfo> &BlockInfo);

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
  LocalVariableMap* VMap;
  LocalVariableMap::Context Ctx;

  VarMapBuilder(LocalVariableMap *VM, LocalVariableMap::Context C)
      : VMap(VM), Ctx(C) {}

  void VisitDeclStmt(const DeclStmt *S);
  void VisitBinaryOperator(const BinaryOperator *BO);
  void VisitCallExpr(const CallExpr *CE);

private:
  void clearCallMutations(const CallExpr *CE);
};

} // namespace

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
        modifiedCtx = true;
      }
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

// The context after a call: its non-const reference and pointer arguments'
// definitions cleared; then try-acquire capabilities added back.
void VarMapBuilder::VisitCallExpr(const CallExpr *CE) {
  clearCallMutations(CE);
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
    } else if (getCanonicalDefinitionID(P.second) !=
               getCanonicalDefinitionID(*I2)) {
      // If canonical definitions mismatch the underlying definitions are
      // different, invalidate.
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
    assert(VDef->isReference());

    const unsigned *I2 = C2.lookup(P.first);
    if (!I2) {
      // Variable does not exist at the end of the loop, invalidate.
      VDef->invalidateRef();
      continue;
    }

    // Compare the canonical IDs. This correctly handles chains of references
    // and determines if the variable is truly loop-invariant.
    if (VDef->CanonicalRef != getCanonicalDefinitionID(*I2))
      VDef->invalidateRef(); // Mark this variable as undefined
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
void LocalVariableMap::traverseCFG(CFG *CFGraph,
                                   const PostOrderCFGView *SortedGraph,
                                   std::vector<CFGBlockInfo> &BlockInfo) {
  PostOrderCFGView::CFGBlockSet VisitedBlocks(CFGraph);

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
    VarMapBuilder VMapBuilder(this, CurrBlockInfo->EntryContext);
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
/// case labels, whether the listed cases cover zero and one, and whether
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
  void recordTryAcquireCalls();
  void reconcileTryAcquireCaps(TryAcquireCaps &Caps);

  /// What decodeTrylockCond()'s walk over a branched-on condition has learned
  /// about the expressions applied to the try-acquire call's result.
  struct TrylockDecode {
    /// The try-acquire call the walk reached, or null if it found none.
    const CallExpr *TrylockCall = nullptr;
    /// The condition tests the negation of the call's result.
    bool Negate = false;
  };

  void decodeTrylockCond(const Stmt *Cond, LocalVarContext C, TrylockDecode &D);

  // Memoize per-switch summaries (resolveTrylockEdge() visits a switch
  // once per successor).
  llvm::SmallDenseMap<const SwitchStmt *, SwitchSummary, 4> SwitchSummaries;
  const SwitchSummary &getSwitchSummary(ASTContext &Ctx, const SwitchStmt *SW);
  const TrylockBranch &decodeTrylockBranch(const CFGBlock *Block);

  /// The try-acquire call a condition branches on, found through its
  /// short-circuit blocks (getConditionTrylockCallExpr()).
  struct ConditionTrylockCall {
    const CallExpr *TrylockCall = nullptr;
    /// Whether every outgoing path of the block reaches the branch on the
    /// call's result; computed on request only.
    bool ResolvesAllPaths = true;
  };
  ConditionTrylockCall getConditionTrylockCallExpr(const CFGBlock *Block,
                                                   bool CheckAllPaths = false);

  /// What one edge proves about the capabilities of a TrylockBranch.
  struct TrylockEdge {
    const CallExpr *TrylockCall = nullptr;
    /// The edge cannot be taken at all (e.g. the implicit default of a
    /// switch that lists every value of a boolean condition).
    bool Infeasible = false;
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

static bool getStaticBooleanValue(Expr *E, bool &TCond) {
  if (isa<CXXNullPtrLiteralExpr>(E) || isa<GNUNullExpr>(E)) {
    TCond = false;
    return true;
  } else if (const auto *BLE = dyn_cast<CXXBoolLiteralExpr>(E)) {
    TCond = BLE->getValue();
    return true;
  } else if (const auto *ILE = dyn_cast<IntegerLiteral>(E)) {
    TCond = ILE->getValue().getBoolValue();
    return true;
  } else if (auto *CE = dyn_cast<ImplicitCastExpr>(E))
    return getStaticBooleanValue(CE->getSubExpr(), TCond);
  return false;
}

// If Cond can be traced back to a try-acquire function call, the `D` variable
// will be populated with the call and with how the branched-on value relates
// to its result.
void ThreadSafetyAnalyzer::decodeTrylockCond(const Stmt *Cond,
                                             LocalVarContext C,
                                             TrylockDecode &D) {
  if (!Cond)
    return;

  if (const auto *CallExp = dyn_cast<CallExpr>(Cond)) {
    if (CallExp->getBuiltinCallee() == Builtin::BI__builtin_expect)
      return decodeTrylockCond(CallExp->getArg(0), C, D);
    const auto *FD = dyn_cast_or_null<NamedDecl>(CallExp->getCalleeDecl());
    if (FD && FD->hasAttr<TryAcquireCapabilityAttr>())
      D.TrylockCall = CallExp;
    return;
  }
  else if (const auto *PE = dyn_cast<ParenExpr>(Cond))
    return decodeTrylockCond(PE->getSubExpr(), C, D);
  else if (const auto *CE = dyn_cast<ImplicitCastExpr>(Cond))
    return decodeTrylockCond(CE->getSubExpr(), C, D);
  else if (const auto *FE = dyn_cast<FullExpr>(Cond))
    return decodeTrylockCond(FE->getSubExpr(), C, D);
  else if (const auto *DRE = dyn_cast<DeclRefExpr>(Cond)) {
    const Expr *E = LocalVarMap.lookupExpr(DRE->getDecl(), C);
    return decodeTrylockCond(E, C, D);
  }
  else if (const auto *UOP = dyn_cast<UnaryOperator>(Cond)) {
    if (UOP->getOpcode() == UO_LNot) {
      D.Negate = !D.Negate;
      return decodeTrylockCond(UOP->getSubExpr(), C, D);
    }
    return;
  }
  else if (const auto *BOP = dyn_cast<BinaryOperator>(Cond)) {
    if (BOP->getOpcode() == BO_EQ || BOP->getOpcode() == BO_NE) {
      if (BOP->getOpcode() == BO_NE)
        D.Negate = !D.Negate;

      bool TCond = false;
      if (getStaticBooleanValue(BOP->getRHS(), TCond)) {
        if (!TCond)
          D.Negate = !D.Negate;
        return decodeTrylockCond(BOP->getLHS(), C, D);
      }
      TCond = false;
      if (getStaticBooleanValue(BOP->getLHS(), TCond)) {
        if (!TCond)
          D.Negate = !D.Negate;
        return decodeTrylockCond(BOP->getRHS(), C, D);
      }
      return;
    }
    if (BOP->getOpcode() == BO_LAnd) {
      // LHS must have been evaluated in a different block.
      return decodeTrylockCond(BOP->getRHS(), C, D);
    }
    if (BOP->getOpcode() == BO_LOr)
      return decodeTrylockCond(BOP->getRHS(), C, D);
    // An assignment used as a condition (`if ((b = mu.TryLock()))`)
    // evaluates to its right-hand side.
    if (BOP->getOpcode() == BO_Assign)
      return decodeTrylockCond(BOP->getRHS(), C, D);
    return;
  } else if (const auto *COP = dyn_cast<ConditionalOperator>(Cond)) {
    bool TCond, FCond;
    if (getStaticBooleanValue(COP->getTrueExpr(), TCond) &&
        getStaticBooleanValue(COP->getFalseExpr(), FCond)) {
      if (TCond && !FCond)
        return decodeTrylockCond(COP->getCond(), C, D);
      if (!TCond && FCond) {
        D.Negate = !D.Negate;
        return decodeTrylockCond(COP->getCond(), C, D);
      }
    }
  } else if (const auto *SE = dyn_cast<StmtExpr>(Cond)) {
    if (const auto *CS = SE->getSubStmt(); CS && !CS->body_empty()) {
      if (const auto *E = dyn_cast<Expr>(CS->body_back()))
        return decodeTrylockCond(E, C, D);
    }
  }
}

/// Find the try-acquire call whose result the condition starting at
/// \p Block branches on. Unlike the plain terminator decode, this looks
/// through short-circuit evaluation: in a compound condition such as
/// `if (c && ok)`, \p Block tests only `c` and the branch on the
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
  // The walk follows only the successor edges of logical-operator
  // terminators, which stay within one condition expression: a back edge
  // originates only from a loop or goto terminator, so the walk cannot
  // cycle and is linear in the size of the condition. The visited set is
  // shared with the escape walks below: a walk that reaches an
  // already-visited block has merged into a path already verified to reach
  // the call (any walk that fails ends the search).
  llvm::SmallPtrSet<const CFGBlock *, 8> Visited;
  SmallVector<const CFGBlock *, 4> Escapes;
  // The blocks whose terminator decodes to the call: their outgoing edges
  // are what must actually resolve the demoted fact.
  SmallVector<const CFGBlock *, 2> Deciders;
  auto Walk = [&](const CFGBlock *Block) -> const CallExpr * {
    while (Block) {
      if (!Visited.insert(Block).second) {
        const CallExpr *Exp = decodeTrylockBranch(Block).TrylockCall;
        if (Exp)
          Deciders.push_back(Block);
        return Exp;
      }
      if (const CallExpr *Exp = decodeTrylockBranch(Block).TrylockCall) {
        Deciders.push_back(Block);
        return Exp;
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
      return nullptr;
    }
    return nullptr;
  };

  ConditionTrylockCall Result;
  const CallExpr *Exp = Result.TrylockCall = Walk(Block);
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
    // Finding the branch is not enough: it has to decide the result on at
    // least one edge. A switch on the result whose labels pin nothing
    // (`switch (ok) { default: }`) leaves every edge Unknown, so
    // getEdgeLockset() resolves nothing and a silently demoted hold would
    // leak unreported. A branch that resolves some of its edges keeps the
    // promise where it can; what escapes the others is the ordinary
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
          AnyEdgeResolves = Edge.Infeasible || Edge.TrylockCall;
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
  return BrE && !BrE->isValueDependent() &&
         BrE->EvaluateAsBooleanCondition(Result, Ctx) && Result;
}

/// The exact result code a try-acquire's attributes declare for success,
/// when they declare one other than 1: the call then reports success by
/// that value and not by truthiness, so a switch that pins some other
/// nonzero value is a failure a truthiness reading takes for a success.
/// Returns whether \p D declares such a code at all; \p Code is the code
/// itself, and is left empty when several attributes declare different
/// ones -- no single label proves success then. A code of 1, or a plain
/// `true`, is ordinary truthiness and reports nothing here, as does the
/// falsy convention (a zero success value), whose edges the truthiness
/// reading already decides exactly.
static bool declaresExactSuccessCode(const NamedDecl *D, ASTContext &Ctx,
                                     std::optional<llvm::APSInt> &Code) {
  bool Exact = false, Conflict = false;
  Code.reset();
  for (const Attr *At : D->attrs()) {
    const auto *A = dyn_cast<TryAcquireCapabilityAttr>(At);
    if (!A || !getTrySuccessValue(Ctx, A->getSuccessValue()))
      continue;
    std::optional<llvm::APSInt> V =
        A->getSuccessValue()->getIntegerConstantExpr(Ctx);
    if (!V || *V == 1)
      continue;
    Exact = true;
    if (Code && llvm::APSInt::compareValues(*Code, *V) != 0)
      Conflict = true;
    Code = V;
  }
  if (Conflict)
    Code.reset();
  return Exact;
}

/// Whether the edge to \p CaseBlock is a case label of \p Sum's switch
/// pinning exactly the value \p Code.
static bool switchEdgePins(const SwitchSummary &Sum, const CFGBlock *CaseBlock,
                           const llvm::APSInt &Code) {
  const auto *CS = dyn_cast_if_present<CaseStmt>(CaseBlock->getLabel());
  if (!CS)
    return false;
  auto It = Sum.OwnCases.find(CS);
  if (It == Sum.OwnCases.end())
    return false;
  const auto &[Lo, Hi] = It->second;
  return llvm::APSInt::compareValues(Lo, Code) == 0 &&
         llvm::APSInt::compareValues(Hi, Code) == 0;
}

/// Decode what the terminator of \p Block branches on: if it is the result of
/// a call to a function annotated with try_acquire_capability (possibly
/// negated, or stored in a local variable), return that call together with the
/// resolution each branch direction proves for each capability recorded for
/// it. Attributes may carry different success values, so each is recorded on
/// its own. The decode is memoized in the block's CFGBlockInfo, and the
/// returned reference stays valid for the rest of the analysis.
const TrylockBranch &
ThreadSafetyAnalyzer::decodeTrylockBranch(const CFGBlock *Block) {
  const unsigned BlockID = Block->getBlockID();

  std::optional<TrylockBranch> &Memo = BlockInfo[BlockID].TryBranch;
  if (Memo)
    return *Memo;
  auto CacheMiss = [&]() -> const TrylockBranch & { return Memo.emplace(); };

  const Stmt *Cond = Block->getTerminatorCondition();
  if (!Cond)
    return CacheMiss();

  // We don't acquire try-locks on ?: branches, except when its result is used.
  if (const auto *COp =
          dyn_cast_if_present<ConditionalOperator>(Block->getTerminatorStmt()))
    if (!COp->getType()->isVoidType())
      return CacheMiss();

  TrylockDecode D;
  decodeTrylockCond(Cond, BlockInfo[BlockID].ExitContext, D);
  if (!D.TrylockCall)
    return CacheMiss();

  // Translate call truthiness to branch truthiness.
  TrylockBranch Result;
  Result.TrylockCall = D.TrylockCall;
  if (auto MapIt = TryAcquireCapsMap.find(D.TrylockCall);
      MapIt != TryAcquireCapsMap.end()) {
    const TryAcquireCaps &Caps = MapIt->second;
    auto AddCaps = [&](const CapExprSet &CapSet, LockKind LK, bool Success) {
      for (const CapabilityExpr &CE : CapSet) {
        (Success != D.Negate ? Result.OnTrue : Result.OnFalse)
            .push_back({CE, LK, CapResolution::Success});
        (Success != D.Negate ? Result.OnFalse : Result.OnTrue)
            .push_back({CE, LK, CapResolution::Failure});
      }
    };
    AddCaps(Caps.TruthyExclusive, LK_Exclusive, /*Success=*/true);
    AddCaps(Caps.TruthyShared, LK_Shared, /*Success=*/true);
    AddCaps(Caps.FalsyExclusive, LK_Exclusive, /*Success=*/false);
    AddCaps(Caps.FalsyShared, LK_Shared, /*Success=*/false);
  }
  // A fully-reconciled call (every capability moved to the unconditional
  // groups) records nothing here: it creates no try facts, and a branch
  // on its result proves nothing.
  if (Result.OnTrue.empty() && Result.OnFalse.empty())
    return CacheMiss();
  return Memo.emplace(std::move(Result));
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
/// \p CaseBlock.
static EdgeValue getSwitchEdgeValue(const SwitchSummary &Sum,
                                    const CFGBlock *CaseBlock) {
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
    return EdgeValue::True;
  }

  // The default edge (explicit, or the implicit fall-out successor): the
  // value matches none of the case labels. If zero is listed the value must
  // be nonzero; for a boolean condition with one listed it must be zero --
  // and with both listed this edge cannot be taken at all.
  if (Sum.ZeroListed)
    return Sum.IsBool && Sum.OneListed ? EdgeValue::Infeasible
                                       : EdgeValue::True;
  if (Sum.IsBool && Sum.OneListed)
    return EdgeValue::False;
  return EdgeValue::Unknown;
}

/// Decode what the edge from \p PredBlock to \p CurrBlock proves about
/// conditional capabilities, selected by the truthiness the edge assigns
/// to the branched-on value. An edge that does not determine the value
/// reports no branch at all: the try facts stay untouched either way.
ThreadSafetyAnalyzer::TrylockEdge
ThreadSafetyAnalyzer::resolveTrylockEdge(const CFGBlock *PredBlock,
                                         const CFGBlock *CurrBlock) {
  const TrylockBranch &B = decodeTrylockBranch(PredBlock);
  TrylockEdge Edge;
  if (!B.TrylockCall)
    return Edge;

  // Determine the truthiness of the branched-on value along this edge.
  EdgeValue CondVal = EdgeValue::Unknown;
  if (const auto *SW =
          dyn_cast_if_present<SwitchStmt>(PredBlock->getTerminatorStmt())) {
    const NamedDecl *Callee = cast<NamedDecl>(B.TrylockCall->getCalleeDecl());
    ASTContext &Ctx = Callee->getASTContext();
    const SwitchSummary &Sum = getSwitchSummary(Ctx, SW);
    CondVal = getSwitchEdgeValue(Sum, CurrBlock);
    // A switch reads the result itself, not its truthiness, so an exact
    // success code has to be matched exactly: every other nonzero edge --
    // another case label, or a default that only proves the value nonzero
    // -- is a failure the truthiness reading would take for a success.
    // Only a label pinning the code proves success here; the rest decide
    // nothing until the analysis resolves values (a commit above). A zero
    // edge is a failure either way and keeps its resolution.
    if (CondVal == EdgeValue::True) {
      std::optional<llvm::APSInt> Code;
      if (declaresExactSuccessCode(Callee, Ctx, Code) &&
          !(Code && switchEdgePins(Sum, CurrBlock, *Code)))
        CondVal = EdgeValue::Unknown;
    }
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
  if (CondVal == EdgeValue::Unknown)
    return Edge;

  Edge.TrylockCall = B.TrylockCall;
  const SmallVectorImpl<TrylockEdgeCap> &Dir =
      CondVal == EdgeValue::True ? B.OnTrue : B.OnFalse;
  Edge.Caps.assign(Dir.begin(), Dir.end());
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

  // Whether a capability is acquired on this edge: it is re-identified by
  // matching against the capabilities recorded at the call, with the
  // resolution this edge proves for each (resolveTrylockEdge()).
  auto FactSucceedsHere = [&](const CapabilityExpr &FE) {
    assert(!Edge.Caps.empty() &&
           "try-acquire fact without capabilities recorded at its call");
    if (llvm::any_of(Edge.Caps, [&](const TrylockEdgeCap &EC) {
          return EC.Resolution == CapResolution::Success && FE.matches(EC.Cap);
        }))
      return true;
    if (llvm::any_of(Edge.Caps, [&](const TrylockEdgeCap &EC) {
          return FE.matches(EC.Cap);
        }))
      return false;
    // A hold of the capability a release-style try-acquire gives up
    // (try_acquire_capability(true, !mu) named !mu, this try fact is mu's):
    // the call's outcome resolves it inverted -- acquiring !mu releases mu,
    // failing to acquire it leaves the hold standing. A join demotes such
    // a hold to the call's try fact (intersectAndWarn()).
    CapabilityExpr Inverse = !FE;
    const auto *InvEC = llvm::find_if(Edge.Caps, [&](const TrylockEdgeCap &EC) {
      return Inverse.matches(EC.Cap);
    });
    assert(InvEC != Edge.Caps.end() &&
           "try-acquire fact matches neither polarity's capabilities");
    return InvEC != Edge.Caps.end() &&
           InvEC->Resolution == CapResolution::Failure;
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
  bool Infeasible = false;
  for (const auto &Fact : Result) {
    const auto *W = dyn_cast<TryFactEntry>(&FactMan[Fact]);
    if (!W || W->origin() != Exp)
      continue;
    switch (W->state()) {
    case TryFactEntry::State::Conditional:
    case TryFactEntry::State::Released:
      Resolvable.push_back(W);
      break;
    case TryFactEntry::State::ProvedHeld:
      // The hold proved by this call's success: an edge on which the
      // capability's own attribute reports failure cannot be taken.
      if (!FactSucceedsHere(*W))
        Infeasible = true;
      break;
    case TryFactEntry::State::ProvedNotHeld:
      // The call provably failed to acquire this try fact's capability on
      // every path here, so an edge on which the capability's own
      // attribute reports success cannot be taken; any other edge is
      // simply consistent with it (attributes carry their own success
      // values, so the test is per try fact, not per edge).
      if (FactSucceedsHere(*W))
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
    const bool Succeeds = FactSucceedsHere(*W);
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
      Result.replaceFact(
          FactMan, *W,
          W->withState(FactMan, TryFactEntry::State::ProvedNotHeld));
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
  // release already discharged a level.
  if (auto MapIt = TryAcquireCapsMap.find(Exp);
      MapIt != TryAcquireCapsMap.end() && !FactMan.spentTryAcquire(Exp)) {
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
    // (tryheld_retry_with_continue). A fresh execution records afresh.
    TryCaps->TrackedCaps.clear();
    // A scoped object manages the try facts its construction created, and
    // only those, each once whatever its kinds: a capability the construction
    // also acquires definitely is managed by that acquisition, and one whose
    // try fact addTryLock() declined was never acquired conditionally, so the
    // destructor has nothing of its to release.
    auto AddTry = [&](const CapExprSet &CapSet, LockKind LK) {
      for (const CapabilityExpr &M : CapSet) {
        if (ExclusiveLocksToAdd.contains(M) || SharedLocksToAdd.contains(M))
          continue;
        if (Analyzer->addTryLock(FSet, M, LK, Loc, Exp, Source)) {
          TryCaps->TrackedCaps.emplace_back(M, LK);
          TryLocksManaged.push_back_nodup(M);
        }
      }
    };
    AddTry(TryCaps->TruthyExclusive, LK_Exclusive);
    AddTry(TryCaps->FalsyExclusive, LK_Exclusive);
    AddTry(TryCaps->TruthyShared, LK_Shared);
    AddTry(TryCaps->FalsyShared, LK_Shared);
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
  bool isTrylockRebranched(const TryFactEntry &W) const {
    return Ctx.RebranchTryLock && W.origin() == Ctx.RebranchTryLock;
  }
  const TryFactEntry *rebranchProof(const FactSet &Set,
                                    const FactEntry &Def) const;
  bool isRebranchedInverseHold(const FactSet &OtherSet,
                               const FactEntry &FE) const;
  /// Either form of the rebranch exemption for the hold \p FE of the
  /// side \p OwnSet, against the other side of the join.
  bool rebranchExemptAgainst(const FactSet &OwnSet, const FactSet &OtherSet,
                             const FactEntry &FE) const {
    return rebranchProof(OwnSet, FE) || isRebranchedInverseHold(OtherSet, FE);
  }
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
  bool holdDemotable(const FactSet &OwnSet, const FactSet &OtherSet,
                     const FactEntry &FE) const {
    return rebranchExemptAgainst(OwnSet, OtherSet, FE) &&
           !rebranchVetoedByReleased(OtherSet, FE);
  }
  bool mixedJoinExempt(const FactSet &DefSet, const FactEntry &DefSide,
                       const FactSet &CondSet) const;
  bool conditionalKeptAgainst(const TryFactEntry &W, const FactSet &OtherSet,
                              bool AllowRebranch = true) const;
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
  const FactEntry *demoteToConditional(const FactEntry &Def, LockErrorKind LEK);
  void demoteExitFact(const FactEntry &Def, LockErrorKind LEK);
  void demoteEntryFactInPlace(const FactEntry &Def, LockErrorKind LEK);
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
    return W.provedHeld() && W.origin() == Ctx.RebranchTryLock;
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
  return OtherSet.findTryFactFrom(FactMan, FE, Ctx.RebranchTryLock,
                                  TryFactEntry::State::Released);
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

// The mixed join is forgiven when the definite side's extra level was
// proved by the call the terminator rebranches on and the other side
// holds that call's conditional try fact: the merged state re-resolves it.
bool LocksetJoin::mixedJoinExempt(const FactSet &DefSet,
                                  const FactEntry &DefSide,
                                  const FactSet &CondSet) const {
  if (!rebranchProof(DefSet, DefSide) || !Ctx.RebranchResolvesAllPaths)
    return false;
  const TryFactEntry *W = CondSet.findTryFact(
      FactMan, DefSide, Ctx.RebranchTryLock, DefSide.kind());
  return W && W->conditional();
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
  if (!Handler.issueBetaWarnings() || !W.lossNeedsWarning())
    return;
  // A capability managed by a scoped object is exempt at an interior
  // join, as for the lost-hold diagnostics above: the scoped fact is
  // still live and its destructor discharges the conditional acquisition.
  // Only at the end of the function, where the guard may have escaped its
  // scope, can the guard fail to check it.
  assert(LEK != LEK_LockedSomeLoopIterations &&
         "the exemption at a loop join is the caller's to make");
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

// Demote the definite hold \p Def, proved by the rebranched call, to that
// call's conditional try fact in the entry set, which a branch on the
// result resolves. A mismatched reentrancy depth is diagnosed here but
// kept -- after the warning, the deeper fact guards more of the releases
// downstream than a stripped one would -- as the levels below the demoted
// one, no longer proved by any call: returned for the caller to place,
// since \p Def itself may belong to the other side.
const FactEntry *LocksetJoin::demoteToConditional(const FactEntry &Def,
                                                  LockErrorKind LEK) {
  const auto &LDef = cast<LockableFactEntry>(Def);
  const Expr *Origin = Ctx.RebranchTryLock;
  if (LDef.getReentrancyDepth() != 0)
    warnReentrancyMismatch(Def, LEK);
  // The call's try fact in the merged set is conditional: the proof it
  // gave on its side does not hold on the other.
  if (FactSet::iterator It =
          EntrySet.findTryFactIter(FactMan, Def, Origin, Def.kind());
      It != EntrySet.end()) {
    const auto &W = cast<TryFactEntry>(FactMan[*It]);
    if (!W.conditional())
      EntrySet.replaceFact(
          FactMan, It, W.withState(FactMan, TryFactEntry::State::Conditional));
  } else {
    EntrySet.addLock(FactMan, LDef.asConditional(FactMan, Origin));
  }
  return LDef.leaveReentrant(FactMan);
}

// Demote the exit set's hold \p Def into the entry set, adding the levels
// below it beside the conditional try fact.
void LocksetJoin::demoteExitFact(const FactEntry &Def, LockErrorKind LEK) {
  if (const FactEntry *Rest = demoteToConditional(Def, LEK))
    EntrySet.addLock(FactMan, Rest);
}

// Demote the entry set's own hold \p Def in place: the levels below it
// take its slot, or it is removed.
void LocksetJoin::demoteEntryFactInPlace(const FactEntry &Def,
                                         LockErrorKind LEK) {
  if (const FactEntry *Rest = demoteToConditional(Def, LEK))
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
  if (!Ctx.canModify())
    return;
  if (EntryW.provedNotHeld() || ExitW.provedNotHeld()) {
    if (CondSide && conditionalKeptAgainst(*CondSide, CondOther)) {
      // The conditional side, may-be-released or not, is the merged try fact.
      if (CondSide != &EntryW)
        EntrySet.replaceFact(FactMan, EntryIt, CondSide);
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
  // or demotes the proved hold.
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
// after can check it. Loop joins are exempt for now -- a result the code checks
// on the paths around the loop is not a leak, and telling those apart from a
// result never checked anywhere takes resolution of stored results, introduced
// separately. A ProvedHeld try fact is never carried one-sided: the merged
// hold, if any, is not proved by its call, and the definite join demotes it to
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
  // A loop join is exempt for now: the result may be checked on the paths
  // around the loop, which this join alone cannot see. (A commit above
  // narrows the exemption to the results the loop really checks.) A loop
  // join also may not rewrite the entry set, which is why nothing below
  // runs for one.
  if (Ctx.isLoopJoin())
    return;
  if (Ctx.EntryLEK == LEK_LockedAtEndOfFunction) {
    warnNeverChecked(ExitW, Ctx.EntryLEK);
    return;
  }
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
  // As in joinTryFactFromExit(), a loop join neither reports the loss nor
  // rewrites the set, so it has nothing to look for here.
  if (EntryW.conditional() && Ctx.ExitLEK != LEK_LockedSomeLoopIterations) {
    if (conditionalKeptAgainst(EntryW, ExitSet))
      return;
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
  else if (Analyzer.join(EntryFact, ExitFact, Ctx.JoinLoc, Ctx.EntryLEK))
    *EntryIt = Fact;
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
  if (!mixedJoinExempt(DefSet, DefSide, CondSet)) {
    if (CondSide.lossNeedsWarning() &&
        !(CondSide.managed() && Ctx.EntryLEK == LEK_LockedSomePredecessors))
      warnReentrancyMismatch(CondSide, Ctx.EntryLEK);
  } else if (Ctx.canModify() &&
             depth(&EntryFact, !ExitHasCond) != depth(&ExitFact, ExitHasCond)) {
    warnReentrancyMismatch(ExitFact, Ctx.EntryLEK);
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
  if (holdDemotable(ExitSet, EntrySetOrig, ExitFact)) {
    // Held on this predecessor only, but the terminator rebranches on
    // the try-acquire that proved it (or gives it up): demote it to
    // conditionally held without warning, as getEdgeLockset() will re-resolve
    // it on the outgoing edges.
    if (Ctx.canModify()) {
      // A rebranch behind a short-circuit does not resolve the result on
      // the escaping edge: a definite hold weakened here can leak there,
      // so it is diagnosed at this join after all (the demotion stands,
      // for the paths that do rebranch).
      if (!Ctx.RebranchResolvesAllPaths)
        warnRemovedExitFact(ExitFact);
      demoteExitFact(ExitFact, Ctx.EntryLEK);
    }
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
  if (holdDemotable(EntrySetOrig, ExitSet, EntryFact)) {
    if (Ctx.canModify()) {
      // As above: an escaping short-circuit edge means the weakened
      // definite hold is diagnosed at the join after all.
      if (!Ctx.RebranchResolvesAllPaths)
        warnRemovedEntryFact(EntryFact);
      demoteEntryFactInPlace(EntryFact, Ctx.ExitLEK);
    }
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
// definitely as well: a reentrancy-depth mismatch; keep the deeper state, to
// minimize follow-on warnings.
void LocksetJoin::joinMixedFromExit(FactID Fact, const FactEntry &ExitFact,
                                    bool ExitHasCond) {
  if (!ExitHasCond) {
    if (!mixedJoinExempt(ExitSet, ExitFact, EntrySetOrig))
      warnRemovedExitFact(ExitFact);
    else if (Ctx.canModify() && reentrancyDepth(ExitFact) != 0)
      warnReentrancyMismatch(ExitFact, Ctx.EntryLEK);
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
    warnReentrancyMismatch(EntryFact, Ctx.ExitLEK);
    return;
  }
  if (!mixedJoinExempt(EntrySetOrig, EntryFact, ExitSet))
    warnRemovedEntryFact(EntryFact);
  else if (Ctx.canModify() && reentrancyDepth(EntryFact) != 0)
    warnReentrancyMismatch(EntryFact, Ctx.ExitLEK);
  if (Ctx.ExitLEK == LEK_LockedSomePredecessors)
    EntrySet.removeFact(FactMan, EntryFact);
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
void ThreadSafetyAnalyzer::reconcileTryAcquireCaps(TryAcquireCaps &Caps) {
  // A capability promised on both outcomes is acquired regardless of the
  // result, in the weaker kind if the two outcomes disagree about it.
  for (const CapabilityExpr &M : Caps.TruthyExclusive)
    if (Caps.FalsyExclusive.contains(M))
      Caps.UnconditionalExclusive.push_back_nodup(M);
    else if (Caps.FalsyShared.contains(M))
      Caps.UnconditionalShared.push_back_nodup(M);
  for (const CapabilityExpr &M : Caps.TruthyShared)
    if (!Caps.UnconditionalExclusive.contains(M) &&
        (Caps.FalsyExclusive.contains(M) || Caps.FalsyShared.contains(M)))
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

ThreadSafetyAnalyzer::TryAcquireCaps &
ThreadSafetyAnalyzer::recordTryAcquireCall(const Expr *Exp, const NamedDecl *D,
                                           til::SExpr *Self,
                                           TryAcquireCaps *NoExprCaps) {
  assert((Exp || NoExprCaps) && "expression-less call without a caps store");
  TryAcquireCaps &Caps = Exp ? TryAcquireCapsMap[Exp] : *NoExprCaps;
  for (const Attr *At : D->attrs()) {
    const auto *A = dyn_cast<TryAcquireCapabilityAttr>(At);
    if (!A)
      continue;
    bool Success = getTrySuccessValue(D->getASTContext(), A->getSuccessValue());
    CapExprSet &Group =
        Success ? (A->isShared() ? Caps.TruthyShared : Caps.TruthyExclusive)
                : (A->isShared() ? Caps.FalsyShared : Caps.FalsyExclusive);
    CapExprSet AttrCaps;
    getMutexIDs(AttrCaps, A, Exp, D, Self);
    for (const auto &M : AttrCaps)
      Group.push_back_nodup(M);
  }
  reconcileTryAcquireCaps(Caps);
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
  LocalVarMap.traverseCFG(CFGraph, SortedGraph, BlockInfo);

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
    // is computed lazily on the first join of sets that carry a try fact
    // at all.
    JoinContext Ctx{CurrBlockInfo->EntryLoc, LEK_LockedSomePredecessors,
                    LEK_LockedSomePredecessors};
    bool RebranchTryLockComputed = false;
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
          if (!RebranchTryLockComputed &&
              (HasTryLockFact(CurrBlockInfo->EntrySet) ||
               HasTryLockFact(PrevLockset))) {
            // Compute once; the result depends only on CurrBlock, not on *PI.
            // Skipped entirely (the common case) until some try fact reaches
            // this join.
            const ConditionTrylockCall Call =
                getConditionTrylockCallExpr(CurrBlock, /*CheckAllPaths=*/true);
            Ctx.RebranchTryLock = Call.TrylockCall;
            Ctx.RebranchResolvesAllPaths = Call.ResolvesAllPaths;
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
      JoinContext Ctx{PreLoop->EntryLoc, LEK_LockedSomeLoopIterations,
                      LEK_LockedSomeLoopIterations};
      Ctx.SealedEntry = true;
      intersectAndWarn(PreLoop->EntrySet, LoopEnd->ExitSet, Ctx);
      // A released try fact reaching the loop head on its back edge is
      // evidence that an iteration may have released the capability its
      // call acquired: patch it into the sealed exit sets the loop's exit
      // edges are computed from.
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
