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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
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
  /// Push M onto list, but discard duplicates.
  void push_back_nodup(const CapabilityExpr &CapE) {
    if (llvm::none_of(*this, [=](const CapabilityExpr &CapE2) {
          return CapE.equals(CapE2);
        }))
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
/// try-acquire result was never checked and the capability may be leaked. The
/// fact is dropped here still; a later commit should close that gap.
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
///
/// Two try facts of one identity meet only at a join.
class TryFactEntry final : public FactEntry {
public:
  enum class State : uint8_t { Conditional, ProvedHeld };

private:
  /// The try-acquire call whose result created this fact state.
  const Expr *Origin;

  State St : 8;

  TryFactEntry(const CapabilityExpr &CE, LockKind LK, SourceLocation Loc,
               SourceKind Src, const Expr *Origin)
      : FactEntry(TryFact, CE, LK, Loc, Src), Origin(Origin),
        St(State::Conditional) {
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

  const TryFactEntry *withState(FactManager &FactMan, State S) const;

  /// The definite fact this try fact's success proves: the same
  /// acquisition -- kind, source and location -- now held.
  const LockableFactEntry *asDefinite(FactManager &FactMan) const;

  /// The state two try facts of one identity meet in, one on each side of a
  /// join point: two equal states meet in themselves, and anything else meets
  /// in Conditional, since a proof about one side's paths is no proof about
  /// the other's.
  ///
  /// This is the meet of the states, not the outcome of the join. What the
  /// merged fact becomes, and what happens to the definite facts a
  /// ProvedHeld side proved, is the responsibility of the merge algorithm.
  static State joinStates(State A, State B) {
    if (A == B)
      return A;
    return State::Conditional;
  }

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
};

inline const TryFactEntry *TryFactEntry::withState(FactManager &FactMan,
                                                   State S) const {
  if (S == St)
    return this;
  auto *NewFact = FactMan.createFact<TryFactEntry>(*this);
  NewFact->St = S;
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
      const auto *W = dyn_cast<TryFactEntry>(&FM[ID]);
      return W && W->conditional() && W->origin() == Origin && W->matches(CapE);
    });
    return FactIDs.size() != Before;
  }

  /// Remove every unresolved try fact of \p CapE: the capability is no
  /// longer possibly held through any of them.
  void removeAllConditional(FactManager &FM, const CapabilityExpr &CapE) {
    llvm::erase_if(FactIDs,
                   [&](FactID ID) { return isConditionalOf(FM[ID], CapE); });
  }

  /// The hold of \p CapE is gone: every ProvedHeld try fact of it no longer
  /// proves a live level, and is dropped.
  void releaseProved(FactManager &FM, const CapabilityExpr &CapE) {
    llvm::erase_if(FactIDs, [&](FactID ID) {
      const auto *W = dyn_cast<TryFactEntry>(&FM[ID]);
      return W && W->provedHeld() && W->matches(CapE);
    });
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

  // Save the current context for later replay
  void saveContext(const Stmt *S, Context C) {
    SavedContexts.push_back(std::make_pair(S, C));
  }

  // Adds a new definition to the given context, and returns a new context.
  // This method should be called when declaring a new variable.
  Context addDefinition(const NamedDecl *D, const Expr *Exp, Context Ctx) {
    assert(!Ctx.contains(D));
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
    if (!asserted() && !negative() && !isUniversal()) {
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
    // it no longer proves a live one.
    FSet.releaseProved(FactMan, Cp);
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
/// \p KeepExisting (it already proves as much).
static void installNegativeFact(FactSet &FSet, FactManager &FactMan,
                                const CapabilityExpr &NegCp, SourceLocation Loc,
                                bool KeepExisting) {
  if (KeepExisting && FSet.findDefinite(FactMan, NegCp))
    return;
  FSet.addLock(FactMan,
               FactMan.createFact<LockableFactEntry>(NegCp, LK_Exclusive, Loc));
}

/// The location for an unmatched-unlock "released here" note: the negative
/// fact's location if one exists.
static SourceLocation unmatchedUnlockNoteLoc(const FactSet &FSet,
                                             FactManager &FactMan,
                                             const CapabilityExpr &Cp) {
  if (const FactEntry *Neg = FSet.findDefinite(FactMan, !Cp))
    return Neg->loc();
  return SourceLocation();
}

/// Release the capability \p Cp, which is only conditionally held (conditional
/// try facts but no definite fact); returns true if the release was handled
/// here. Diagnose like an unmatched unlock, drop every conditional try fact,
/// and leave the negative fact behind: the release is an unconditional
/// demand, and the thread provably does not hold the capability afterwards,
/// whether the try-acquires succeeded or failed.
/// With a null \p Handler (a scoped guard's destructor, from
/// FullyRemove=true) the try facts are kept unchanged: they record
/// acquisitions the guard does not own, which the destructor's conditional
/// release cannot pair with.
static bool handleUncheckedConditionalUnlock(FactSet &FSet,
                                             FactManager &FactMan,
                                             const CapabilityExpr &Cp,
                                             SourceLocation UnlockLoc,
                                             ThreadSafetyHandler *Handler) {
  if (!FSet.anyConditional(FactMan, Cp))
    return false;
  if (Handler) {
    Handler->handleUnmatchedUnlock(Cp.getKind(), Cp.toString(), UnlockLoc,
                                   SourceLocation(), true);
    FSet.removeAllConditional(FactMan, Cp);
    // A pre-existing negative fact survives a try-acquire (it is consumed
    // only on the success edge), so do not add a duplicate over it.
    if (!Cp.negative())
      installNegativeFact(FSet, FactMan, !Cp, UnlockLoc,
                          /*KeepExisting=*/true);
  }
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
/// diagnosed where it is lost. A conditional try fact of the other lock kind
/// cannot be a level of this hold and gives way to the definite level.
static void addLockOverConditional(FactSet &FSet, FactManager &FactMan,
                                   const FactEntry *Entry,
                                   const FactEntry &Cond,
                                   ThreadSafetyHandler *Handler) {
  const bool SameKind =
      FSet.conditionalsAllOfKind(FactMan, *Entry, Entry->kind());
  if (Handler &&
      !(Entry->reentrant() && isa<LockableFactEntry>(Entry) && SameKind))
    Handler->handleDoubleLock(Entry->getKind(), Entry->toString(), Cond.loc(),
                              Entry->loc(), /*MaybeHeld=*/true);
  if (!SameKind)
    FSet.removeAllConditional(FactMan, *Entry);
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
    if (LEK == LEK_LockedAtEndOfFunction || LEK == LEK_NotLockedAtEndOfFunction)
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
          Handler);
      return;
    }
    FSet.removeDefinite(FactMan, !Cp);
    FSet.addLock(FactMan,
                 FactMan.createFact<LockableFactEntry>(Cp, kind, loc, Managed));
  }

  void unlock(FactSet &FSet, FactManager &FactMan, const CapabilityExpr &Cp,
              SourceLocation loc, ThreadSafetyHandler *Handler) const {
    if (const auto It = FSet.findDefiniteIter(FactMan, Cp); It != FSet.end()) {
      const auto &Fact = cast<LockableFactEntry>(FactMan[*It]);
      if (const FactEntry *RFact = Fact.leaveReentrant(FactMan)) {
        // This capability remains reentrantly acquired.
        FSet.replaceFact(FactMan, It, RFact);
        return;
      }

      FSet.erase(It);
      // As in LockableFactEntry::handleUnlock(): released -- unless a
      // conditional try fact remains, in which case the capability is now
      // merely conditionally held.
      FSet.releaseProved(FactMan, Cp);
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

  /// A loop join compares a back edge's exit set against an entry set
  /// analyzed previously: it diagnoses, but must not rewrite that set.
  bool isLoopJoin() const { return EntryLEK == LEK_LockedSomeLoopIterations; }
  bool canModify() const { return !isLoopJoin(); }
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
  // Maps constructed objects to `this` placeholder prior to initialization.
  llvm::SmallDenseMap<const Expr *, til::LiteralPtr *> ConstructedObjects;
  /// The capabilities named by a try-acquire call's attributes, translated
  /// in the call's own context and grouped by the attribute's lock kind and
  /// success value (Falsy: reported acquired when the call returns false).
  struct TryAcquireCaps {
    CapExprSet TruthyExclusive, TruthyShared;
    CapExprSet FalsyExclusive, FalsyShared;
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
  void addTryLock(FactSet &FSet, const CapabilityExpr &CE, LockKind LK,
                  SourceLocation Loc, const Expr *Call);
  void checkAcquiredCapability(FactSet &FSet, const FactEntry &Entry,
                               bool ReqAttr);
  void removeLock(FactSet &FSet, const CapabilityExpr &CapE,
                  SourceLocation UnlockLoc, bool FullyRemove, LockKind Kind);

  template <typename AttrType>
  void getMutexIDs(CapExprSet &Mtxs, AttrType *Attr, const Expr *Exp,
                   const NamedDecl *D, til::SExpr *Self = nullptr);

  void recordTryAcquireCall(const Expr *Exp, const NamedDecl *D,
                            til::SExpr *Self = nullptr);
  void recordTryAcquireCalls();

  /// What decodeTrylockCond()'s walk over a branched-on condition has learned
  /// about the expressions applied to the try-acquire call's result.
  struct TrylockDecode {
    /// The try-acquire call the walk reached, or null if it found none.
    const CallExpr *TrylockCall = nullptr;
    /// The condition tests the negation of the call's result.
    bool Negate = false;
  };

  void decodeTrylockCond(const Stmt *Cond, LocalVarContext C, TrylockDecode &D);

  const TrylockBranch &decodeTrylockBranch(const CFGBlock *Block);

  /// What one edge proves about the capabilities of a TrylockBranch.
  struct TrylockEdge {
    const CallExpr *TrylockCall = nullptr;
    SmallVector<TrylockEdgeCap, 2> Caps;
  };
  TrylockEdge resolveTrylockEdge(const CFGBlock *PredBlock,
                                 const CFGBlock *CurrBlock);

  void getEdgeLockset(FactSet &Result, const FactSet &ExitSet,
                      const CFGBlock* PredBlock,
                      const CFGBlock *CurrBlock);

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

/// The checks required before an acquisition: consume (or require) the
/// negative capability, and check acquired_before/acquired_after ordering.
/// A try-acquire attempts the acquisition, so a try fact \p Entry is checked
/// the same way -- once, at the call -- but leaves the negative fact in
/// place: it is consumed on the call's success edge.
void ThreadSafetyAnalyzer::checkAcquiredCapability(FactSet &FSet,
                                                   const FactEntry &Entry,
                                                   bool ReqAttr) {
  if (!ReqAttr && !Entry.negative()) {
    // look for the negative capability, and remove it from the fact set.
    CapabilityExpr NegC = !Entry;
    if (const FactEntry *Nen = FSet.findDefinite(FactMan, NegC)) {
      if (!isa<TryFactEntry>(Entry))
        FSet.removeFact(FactMan, *Nen);
    } else {
      if (inCurrentScope(Entry) && !Entry.asserted() && !Entry.reentrant())
        Handler.handleNegativeNotHeld(Entry.getKind(), Entry.toString(),
                                      NegC.toString(), Entry.loc());
    }
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
void ThreadSafetyAnalyzer::addTryLock(FactSet &FSet, const CapabilityExpr &CE,
                                      LockKind LK, SourceLocation Loc,
                                      const Expr *Call) {
  auto *Fact =
      FactMan.createFact<TryFactEntry>(CE, LK, Loc, FactEntry::Acquired, Call);
  if (Fact->shouldIgnore())
    return;

  checkAcquiredCapability(FSet, *Fact, /*ReqAttr=*/false);

  if (const FactEntry *Cp = FSet.findDefinite(FactMan, CE)) {
    // A try-acquire naming a scoped object -- a guard's own TryLock() member --
    // is trackable: the try fact speaks about the guard, and its success edge
    // replays what the guard acquires, as a blocking member of the guard does.
    if (!isa<ScopedLockableFactEntry>(Cp) && Cp->kind() != LK) {
      Handler.handleDoubleLock(CE.getKind(), CE.toString(), Cp->loc(), Loc,
                               /*MaybeHeld=*/false);
      return;
    }
  }
  SmallVector<const TryFactEntry *, 2> TryFacts;
  FSet.collectTryFacts(FactMan, CE, TryFacts);
  for (const TryFactEntry *W : TryFacts) {
    if (W->conditional()) {
      if (W->kind() != LK || W->origin() == Call) {
        Handler.handleDoubleLock(CE.getKind(), CE.toString(), W->loc(), Loc,
                                 /*MaybeHeld=*/true);
        return;
      }
    } else if (W->origin() == Call && W->kind() == LK) {
      // A fresh execution of the call overwrites its stored result: a hold
      // an earlier execution proved is no longer determined by it, and the
      // try fact starts over as the new result's.
      FSet.removeFact(FactMan, *W);
    }
  }
  FSet.addLock(FactMan, Fact);
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

/// Decode a try-acquire attribute's success value. An expression that does
/// not constant-evaluate reads as false.
static bool getTrySuccessValue(ASTContext &Ctx, const Expr *BrE) {
  bool Result;
  return BrE && !BrE->isValueDependent() &&
         BrE->EvaluateAsBooleanCondition(Result, Ctx) && Result;
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
  return Memo.emplace(std::move(Result));
}

/// Decode what the edge from \p PredBlock to \p CurrBlock proves about
/// conditional capabilities.
ThreadSafetyAnalyzer::TrylockEdge
ThreadSafetyAnalyzer::resolveTrylockEdge(const CFGBlock *PredBlock,
                                         const CFGBlock *CurrBlock) {
  const TrylockBranch &B = decodeTrylockBranch(PredBlock);
  TrylockEdge Edge;
  if (!B.TrylockCall)
    return Edge;

  // Check which positions among PredBlock's first two successors this edge
  // occupies: for `if`, the first is the condition-true edge, the second the
  // condition-false edge.
  bool TrueEdge = false, FalseEdge = false;
  int i = 0;
  for (CFGBlock::const_succ_iterator SI = PredBlock->succ_begin(),
                                     SE = PredBlock->succ_end();
       SI != SE && i < 2; ++SI, ++i)
    if (*SI == CurrBlock)
      (i == 0 ? TrueEdge : FalseEdge) = true;
  // An edge occupying both positions (the branch reaches the same block
  // either way) has no effect.
  if (TrueEdge && FalseEdge)
    return Edge;

  Edge.TrylockCall = B.TrylockCall;
  if (!TrueEdge && !FalseEdge) {
    // An edge occupying neither position (e.g. a switch case) proves no
    // acquisition.
    for (const TrylockEdgeCap &TC : B.OnTrue)
      Edge.Caps.push_back({TC.Cap, TC.Kind, CapResolution::Failure});
    return Edge;
  }
  const SmallVectorImpl<TrylockEdgeCap> &Dir = TrueEdge ? B.OnTrue : B.OnFalse;
  Edge.Caps.assign(Dir.begin(), Dir.end());
  return Edge;
}

/// Find the lockset that holds on the edge between PredBlock
/// and CurrBlock.  The edge set is the exit set of PredBlock (passed
/// as the ExitSet parameter) plus any trylocks, which are conditionally held.
void ThreadSafetyAnalyzer::getEdgeLockset(FactSet &Result,
                                          const FactSet &ExitSet,
                                          const CFGBlock *PredBlock,
                                          const CFGBlock *CurrBlock) {
  Result = ExitSet;

  TrylockEdge Edge = resolveTrylockEdge(PredBlock, CurrBlock);
  if (!Edge.TrylockCall)
    return;

  // Collect the conditional try facts of this call, to resolve on this
  // edge.
  SmallVector<const TryFactEntry *> ResolvedTryFacts;
  for (const auto &Fact : Result) {
    const auto *W = dyn_cast<TryFactEntry>(&FactMan[Fact]);
    if (W && W->conditional() && W->origin() == Edge.TrylockCall)
      ResolvedTryFacts.push_back(W);
  }
  if (ResolvedTryFacts.empty())
    return;
  assert(!Edge.Caps.empty() &&
         "try-acquire fact without capabilities recorded at its call");

  // Whether the try fact's capability is acquired on this edge: the try fact
  // is re-identified by matching its capability against the capabilities
  // recorded at the call, with the resolution this edge proves for each.
  auto FactSucceedsHere = [&](const FactEntry &FE) {
    if (llvm::any_of(Edge.Caps, [&](const TrylockEdgeCap &EC) {
          return EC.Resolution == CapResolution::Success && FE.matches(EC.Cap);
        }))
      return true;
    assert(llvm::any_of(
               Edge.Caps,
               [&](const TrylockEdgeCap &EC) { return FE.matches(EC.Cap); }) &&
           "try-acquire fact matches neither polarity's capabilities");
    return false;
  };

  // Resolve every try fact of the call on this edge: the success edge folds
  // it into the capability's definite fact -- one level deeper, or newly
  // created -- keeps the try fact as the proof of that level, and consumes
  // the negative capability the call could only require; the failure edge
  // drops it.
  for (const TryFactEntry *W : ResolvedTryFacts) {
    if (!FactSucceedsHere(*W)) {
      Result.removeFact(FactMan, *W);
      continue;
    }
    Result.replaceFact(FactMan, *W,
                       W->withState(FactMan, TryFactEntry::State::ProvedHeld));
    const FactEntry *Def = Result.findDefinite(FactMan, *W);
    if (const auto *Scope = dyn_cast_or_null<ScopedLockableFactEntry>(Def)) {
      // The guard's own try-acquire member succeeded here, so the guard now
      // holds what it manages. The scope's fact is not a hold to deepen: it
      // is the guard, which exists on either edge.
      for (const UnderlyingCapability &UM : Scope->getManaged())
        if (UM.Kind == UCK_Acquired)
          addLock(Result, FactMan.createFact<LockableFactEntry>(
                              UM.Cap, W->kind(), W->loc(), FactEntry::Managed));
    } else if (Def) {
      Result.replaceFact(FactMan, *Def,
                         cast<LockableFactEntry>(Def)->deepen(FactMan));
    } else {
      Result.addLock(FactMan, W->asDefinite(FactMan));
    }
    if (!W->negative())
      Result.removeDefinite(FactMan, !*W);
  }
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

      // Try-acquired capabilities were already recorded for CallExprs, so only
      // a constructor is recorded here, on its first try-acquire attribute,
      // where its constructed-object placeholder is available.
      // The conditional locks are added to our lockset below, from the recorded
      // capabilities in TryAcquireCapsMap.
      case attr::TryAcquireCapability: {
        if (Exp && (!isa<CXXConstructExpr>(Exp) ||
                    Analyzer->TryAcquireCapsMap.contains(Exp)))
          break;
        auto PostContextForThisScope =
            LVarCtx.switchToContextForScope(DualLocalVarContext::Post);
        Analyzer->recordTryAcquireCall(Exp, D, Self);
        break;
      }

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

  // Add conditional locks.
  // Note that scoped lockables manage their underlying mutexes themselves and
  // are not tracked conditionally.
  if (Exp && Scp.shouldIgnore()) {
    if (auto It = Analyzer->TryAcquireCapsMap.find(Exp);
        It != Analyzer->TryAcquireCapsMap.end()) {
      const ThreadSafetyAnalyzer::TryAcquireCaps &Caps = It->second;
      for (const auto &M : Caps.TruthyExclusive)
        Analyzer->addTryLock(FSet, M, LK_Exclusive, Loc, Exp);
      for (const auto &M : Caps.FalsyExclusive)
        Analyzer->addTryLock(FSet, M, LK_Exclusive, Loc, Exp);
      for (const auto &M : Caps.TruthyShared)
        Analyzer->addTryLock(FSet, M, LK_Shared, Loc, Exp);
      for (const auto &M : Caps.FalsyShared)
        Analyzer->addTryLock(FSet, M, LK_Shared, Loc, Exp);
    }
  }

  if (!Scp.shouldIgnore()) {
    // Add the managing object as a dummy mutex, mapped to the underlying mutex.
    auto *ScopedEntry = Analyzer->FactMan.createFact<ScopedLockableFactEntry>(
        Scp, Loc, FactEntry::Acquired,
        ExclusiveLocksToAdd.size() + SharedLocksToAdd.size() +
            ScopedReqsAndExcludes.size() + ExclusiveLocksToRemove.size() +
            SharedLocksToRemove.size());
    for (const auto &M : ExclusiveLocksToAdd)
      ScopedEntry->addLock(M);
    for (const auto &M : SharedLocksToAdd)
      ScopedEntry->addLock(M);
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
  bool mixedJoinExempt(const FactSet &DefSet, const FactEntry &DefSide,
                       const FactSet &CondSet) const;
  /// \}

  /// \name Diagnostics
  /// \{
  void warnRemovedEntryFact(const FactEntry &EntryFact) const;
  void warnRemovedExitFact(const FactEntry &ExitFact) const;
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
  const TryFactEntry *W =
      Set.findTryFact(FactMan, Def, Ctx.RebranchTryLock, Def.kind());
  return W && W->provedHeld() ? W : nullptr;
}

// The mixed join is forgiven when the definite side's extra level was
// proved by the call the terminator rebranches on and the other side
// holds that call's conditional try fact: the merged state re-resolves it.
bool LocksetJoin::mixedJoinExempt(const FactSet &DefSet,
                                  const FactEntry &DefSide,
                                  const FactSet &CondSet) const {
  if (!rebranchProof(DefSet, DefSide))
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

// Diagnose a join where only the guaranteed depth of the hold differs.
void LocksetJoin::warnReentrancyMismatch(const FactEntry &FE,
                                         LockErrorKind LEK) const {
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

// Two try facts of one identity: the merged state is their join.
void LocksetJoin::joinTryFactPair(FactSet::iterator EntryIt,
                                  const TryFactEntry &ExitW) {
  const auto &EntryW = cast<TryFactEntry>(FactMan[*EntryIt]);
  const TryFactEntry::State Merged =
      TryFactEntry::joinStates(EntryW.state(), ExitW.state());
  if (Merged != EntryW.state() && Ctx.canModify())
    EntrySet.replaceFact(FactMan, EntryIt, EntryW.withState(FactMan, Merged));
}

// A try fact of the exit set without a counterpart. A conditional one is
// carried into the merged state whenever the other side holds the capability in
// any form (the other side's extra is what gets diagnosed), or the terminator
// rebranches on its call. Missing from a path that does not hold the capability
// at all, the analysis loses track of it: this predecessor carries a
// try-acquire result into the join (or to the end of the function) without its
// result having been checked. A ProvedHeld try fact is never carried one-sided:
// the merged hold, if any, is not proved by its call, and the definite join
// demotes it to conditional where the rebranch exemption applies.
void LocksetJoin::joinTryFactFromExit(FactID Fact, const TryFactEntry &ExitW) {
  if (!ExitW.conditional() || !Ctx.canModify())
    return;
  if (EntrySetOrig.findDefinite(FactMan, ExitW) ||
      EntrySetOrig.anyConditional(FactMan, ExitW) || isTrylockRebranched(ExitW))
    EntrySet.addLockByID(Fact);
}

// A try fact of the entry set without a counterpart: as above, a conditional
// one is kept wherever the other side holds the capability in any form or the
// terminator rebranches on its call, and lost otherwise -- with the unchecked
// try-acquire on an earlier predecessor. A one-sided ProvedHeld try fact proves
// nothing on the other path and is dropped (unless the definite join already
// turned it conditional, in which case it is no longer here to drop).
void LocksetJoin::joinTryFactFromEntry(const TryFactEntry &EntryW) {
  if (Ctx.ExitLEK != LEK_LockedSomePredecessors)
    return; // Only a branch join rewrites the entry set.
  if (EntryW.conditional() &&
      (ExitSet.findDefinite(FactMan, EntryW) ||
       ExitSet.anyConditional(FactMan, EntryW) || isTrylockRebranched(EntryW)))
    return;
  EntrySet.removeFact(FactMan, EntryW);
}

// Two definite facts of a capability: a mixed join if exactly one side also
// holds it conditionally, else joined. A ProvedHeld try fact on one side only
// is dropped by the try fact joins: the merged hold is not proved by that call.
void LocksetJoin::joinDefinitePair(FactSet::iterator EntryIt, FactID Fact,
                                   const FactEntry &ExitFact) {
  const FactEntry &EntryFact = FactMan[*EntryIt];
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
// fact: the merged state re-resolves it. Otherwise the definite side's fact is
// diagnosed as not held on the other path. The merged state is the conditional
// side's.
void LocksetJoin::joinMixedPair(FactSet::iterator EntryIt, FactID Fact,
                                const FactEntry &EntryFact,
                                const FactEntry &ExitFact, bool ExitHasCond) {
  const FactEntry &DefSide = ExitHasCond ? EntryFact : ExitFact;
  const FactSet &DefSet = ExitHasCond ? EntrySetOrig : ExitSet;
  const FactSet &CondSet = ExitHasCond ? ExitSet : EntrySetOrig;
  if (!mixedJoinExempt(DefSet, DefSide, CondSet)) {
    if (ExitHasCond)
      warnRemovedEntryFact(EntryFact);
    else
      warnRemovedExitFact(ExitFact);
  } else if (Ctx.canModify() &&
             depth(&EntryFact, !ExitHasCond) != depth(&ExitFact, ExitHasCond)) {
    warnReentrancyMismatch(ExitFact, Ctx.EntryLEK);
  }
  if (Ctx.canModify() && ExitHasCond)
    *EntryIt = Fact;
}

// A definite hold of this predecessor only: a mixed join if the other side
// holds the capability conditionally; else demoted to conditional under the
// rebranch exemption; else lost -- silently when its own side also holds the
// capability conditionally (this predecessor carries a try-acquire result into
// the join without its result having been checked), and with the default
// lost-capability diagnostic otherwise.
void LocksetJoin::joinDefiniteFromExit(FactID Fact, const FactEntry &ExitFact) {
  const bool ExitHasCond = ExitSet.anyConditional(FactMan, ExitFact);
  if (EntrySetOrig.anyConditional(FactMan, ExitFact)) {
    joinMixedFromExit(Fact, ExitFact, ExitHasCond);
    return;
  }
  if (rebranchProof(ExitSet, ExitFact)) {
    // Held on this predecessor only, but the terminator rebranches on
    // the try-acquire that proved it: demote it to conditional without
    // warning, as getEdgeLockset() will re-resolve it on the outgoing
    // edges.
    if (Ctx.canModify())
      demoteExitFact(ExitFact, Ctx.EntryLEK);
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
  const bool EntryHasCond = EntrySetOrig.anyConditional(FactMan, EntryFact);
  if (ExitSet.anyConditional(FactMan, EntryFact)) {
    joinMixedFromEntry(EntryFact, EntryHasCond);
    return;
  }
  if (rebranchProof(EntrySetOrig, EntryFact)) {
    if (Ctx.canModify())
      demoteEntryFactInPlace(EntryFact, Ctx.ExitLEK);
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

/// Record the capabilities named by the try-acquire attributes of the call
/// or construction \p Exp to \p D into TryAcquireCapsMap, translated in the
/// currently installed context. Without an expression there is nothing to
/// record or branch on; translate only for the diagnostics.
void ThreadSafetyAnalyzer::recordTryAcquireCall(const Expr *Exp,
                                                const NamedDecl *D,
                                                til::SExpr *Self) {
  TryAcquireCaps DiscardedCaps;
  TryAcquireCaps &Caps = Exp ? TryAcquireCapsMap[Exp] : DiscardedCaps;
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
    // result this block's terminator branches on, if any -- is computed
    // lazily on the first join of sets that carry a try fact at all.
    JoinContext Ctx{CurrBlockInfo->EntryLoc, LEK_LockedSomePredecessors,
                    LEK_LockedSomePredecessors};
    bool RebranchTryLockComputed = false;
    auto HasTryLockFact = [this](const FactSet &FS) {
      // TryAcquireCapsMap is empty in functions without try-acquires (the
      // common case): skip scanning the fact sets entirely.
      return !TryAcquireCapsMap.empty() && llvm::any_of(FS, [this](FactID ID) {
        return isa<TryFactEntry>(FactMan[ID]);
      });
    };
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

      // Okay, we can reach this block from the entry.
      CurrBlockInfo->Reachable = true;

      FactSet PrevLockset;
      getEdgeLockset(PrevLockset, PrevBlockInfo->ExitSet, *PI, CurrBlock);

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
          // edges if the terminator branches on that call's result.
          if (!RebranchTryLockComputed &&
              (HasTryLockFact(CurrBlockInfo->EntrySet) ||
               HasTryLockFact(PrevLockset))) {
            // Compute once; the result depends only on CurrBlock, not on *PI.
            // Skipped entirely (the common case) until some try fact reaches
            // this join.
            Ctx.RebranchTryLock = decodeTrylockBranch(CurrBlock).TrylockCall;
            RebranchTryLockComputed = true;
          }
          intersectAndWarn(CurrBlockInfo->EntrySet, PrevLockset, Ctx);
        }
      }
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
      intersectAndWarn(PreLoop->EntrySet, LoopEnd->ExitSet, PreLoop->EntryLoc,
                       LEK_LockedSomeLoopIterations);
    }
  }

  // Skip the final check if the exit block is unreachable.
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
