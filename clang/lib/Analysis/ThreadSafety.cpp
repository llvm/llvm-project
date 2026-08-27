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
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
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

/// This is a helper class that stores a fact that is known at a
/// particular point in program execution. Concretely, a fact is a capability,
/// along with additional information, such as where it was acquired, whether
/// it is exclusive or shared, etc.
///
/// Per capability the analysis tracks a ternary state: not-held (no fact in
/// the FactSet), try-held, or held. Permitted transitions:
///
///   not-held --acquire-----------------------------------------> held
///   not-held --try-acquire (BuildLockset::handleCall)----------> try-held
///   try-held --branch on the try-acquire result: success edge--> held
///   try-held --branch on the try-acquire result: failure edge--> not-held
///   try-held --acquire or assert (addLock)---------------------> held
///   held -----branch on the originating try-acquire's result:
///             success edge (the failure edge is infeasible
///             and skipped at joins)-----------------------------> held
///   held -----join with a failed path of the same try-acquire,
///             when the join re-branches on its result
///             (intersectAndWarn)-------------------------------> try-held
///   held -----release------------------------------------------> not-held
///
/// A fact additionally tracks a reentrancy depth, and so two of those
/// transitions may deepen that instead of colliding: a try-acquire over
/// a held capability -- whatever its reentrancy, since at runtime such a
/// call fails rather than deadlocks -- and, for a reentrant capability
/// only, an unconditional acquire over a try-held capability. Either way
/// the fact becomes try-held one level deeper with the try-acquire call as
/// its origin: held at that depth if that call succeeded, one level shallower
/// otherwise -- and resolved at a branch on the result and unwound one
/// level per release.
///
/// Branches are resolved in getEdgeLockset(); facts remember their
/// originating call so that later branches on the same result re-resolve
/// them and the join demotion above can identify them. A join of paths
/// holding the capability via different origins clears the merged fact's
/// origin (it is no longer determined by either result).
///
/// Both the join demotion and a branch's fact resolution rest on the
/// premise that a path not holding the capability carries a falsy stored
/// result. Negative facts police that premise: a join keeps a one-sided
/// negative fact as *weak* evidence (not-held on some path), and a
/// release that spends a stored result -- releasing a hold the call's
/// success had proved -- marks its negative fact with the call. A join
/// refuses to demote-and-carry across a spent result, and a branch's
/// success edge re-materializes a fact the analysis lost at a join (e.g.
/// around a loop) only when no surviving negative fact contradicts it.
///
/// Try-held means "held if the try-acquire succeeded", so it warns
/// wherever a definite state is required: it does not satisfy capability
/// requirements, it violates exclusions and negative requirements,
/// releasing it warns (may not be held), and acquiring it again warns
/// (may already be held).
/// Asserts and same-kind reentrant acquires (which deepen instead) are
/// exempt from the acquire warning: they legitimately acquire a
/// possibly-held capability. An acquire of the other kind (shared vs.
/// exclusive) warns even for a reentrant capability: the tracked hold has
/// a single kind.
///
/// When the analysis loses track of a try-held fact -- at a join with a
/// path that does not hold it, or at the end of the function -- the
/// try-acquire result was never checked and the capability may be leaked;
/// this is diagnosed in beta mode.
///
/// Any other try-acquire of a capability that is already tracked is a
/// conflict the model cannot represent: over a try-held fact a second
/// unresolved try-acquire cannot be tracked (one origin per fact),
/// reentrant or not, and over a definite hold a try-acquire of the other
/// kind (shared vs. exclusive) cannot share the fact's single lock kind.
/// Either way the existing fact wins unchanged, the call's acquisition
/// goes untracked, and the conflict is diagnosed at the call.
class FactEntry : public CapabilityExpr {
public:
  enum FactEntryKind { Lockable, ScopedLockable };

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

  /// The try-acquire call this fact originates from (or null), and whether
  /// the capability is still only conditionally ("try") held: acquired by
  /// that call but not yet branched on its result, so held only on the paths
  /// where the call succeeded. Facts promoted to held on the call's success
  /// edge keep their origin (with the flag cleared), so that a later join
  /// with a path where the try-acquire failed can be recognized. Facts
  /// upgraded by an unconditional acquire or assert clear this: their held
  /// state is not proved by the call's success, so a branch on its result
  /// must not resolve them (a reentrant acquire instead deepens the fact,
  /// keeping the origin for its still conditional top level). A try-held
  /// fact whose paths merged different origins loses its origin and can
  /// never be resolved; it stays try-held until the analysis loses track
  /// of it.
  /// (callexpr, true)  : try-held state from callexpr
  /// (callexpr, false) : held state from callexpr
  /// (nullptr, true)   : try-held state from multiple sources
  /// (nullptr, false)  : held state from definitive sources
  llvm::PointerIntPair<const Expr *, 1, bool> TryLock;

  /// Whether this fact holds on only some, not all, paths into the current
  /// program point. Only negative facts are tracked this way: instead of
  /// leaving the intersection silently, a one-sided negative fact is kept
  /// in a join's merged set as a weak fact (intersectAndWarn()) --
  /// evidence that the capability was provably released, or a try-acquire
  /// of it provably failed, on at least one path. The try-held machinery
  /// consults it to refuse carrying (intersectAndWarn()) or
  /// re-materializing (getEdgeLockset()) a hold whose stored try-acquire
  /// result is stale on such a path. A weak fact proves nothing on all
  /// paths: it does not satisfy negative-capability requirements and
  /// cannot prove a branch edge infeasible.
  bool Weak = false;

  /// For a negative fact recorded by the release of a hold that a
  /// try-acquire call's success had proved (a fact promoted from that
  /// call, released by handleUnlock()): that call. The release spends the
  /// call's stored result -- the result stays truthy while the capability
  /// is no longer held -- so a later branch on it must not resurrect the
  /// hold: a join refuses to carry (intersectAndWarn()) and an edge to
  /// re-materialize (getEdgeLockset()) a fact of this call across this
  /// negative. Null for a negative from a plain release or from the call's
  /// failure edge (there the result is provably falsy, and a branch on it
  /// excludes those paths itself). Merges keep it like \c Weak: spent on
  /// some path is spent.
  const Expr *SpentTryLock = nullptr;

protected:
  ~FactEntry() = default;

public:
  FactEntry(FactEntryKind FK, const CapabilityExpr &CE, LockKind LK,
            SourceLocation Loc, SourceKind Src)
      : CapabilityExpr(CE), Kind(FK), LKind(LK), Source(Src), AcquireLoc(Loc) {}

  LockKind kind() const { return LKind;      }
  SourceLocation loc() const { return AcquireLoc; }
  FactEntryKind getFactEntryKind() const { return Kind; }

  bool asserted() const { return Source == Asserted; }
  bool declared() const { return Source == Declared; }
  bool managed() const { return Source == Managed; }

  bool tryHeld() const { return TryLock.getInt(); }
  const Expr *tryLockCall() const { return TryLock.getPointer(); }

  bool weak() const { return Weak; }
  /// Mark this fact as holding on only some paths (see \c Weak).
  void setWeak() { Weak = true; }

  const Expr *spentTryLock() const { return SpentTryLock; }
  /// Record that this negative fact spends \p Call's stored result (see
  /// \c SpentTryLock).
  void setSpentTryLock(const Expr *Call) { SpentTryLock = Call; }

  /// Whether the capability is definitely held at least once: it is not
  /// try-held, or only the top level of a reentrant acquisition is
  /// conditional while the levels below it are definite.
  virtual bool definitelyHeld() const { return !tryHeld(); }

  /// The fact's reentrancy depth; only lockable facts can be reentrant.
  virtual unsigned int getReentrancyDepth() const { return 0; }

  /// Record that this fact originates from the try-acquire call \p Call.
  /// While \p Conditional is true the fact is try-held and is resolved
  /// (promoted to held or removed) at a branch on the call's result; a
  /// try-held fact with a null origin (merged from different origins) can
  /// never be resolved.
  void setTryLock(const Expr *Call, bool Conditional) {
    TryLock.setPointerAndInt(Call, Conditional);
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

/// A FactSet is the set of facts that are known to be true at a
/// particular program point.  FactSets must be small, because they are
/// frequently copied, and are thus implemented as a set of indices into a
/// table maintained by a FactManager.  A typical FactSet only holds 1 or 2
/// locks, so we can get away with doing a linear search for lookup.  Note
/// that a hashtable or map is inappropriate in this case, because lookups
/// may involve partial pattern matches, rather than exact matches.
class FactSet {
private:
  using FactVec = SmallVector<FactID, 4>;

  FactVec FactIDs;

public:
  using iterator = FactVec::iterator;
  using const_iterator = FactVec::const_iterator;

  iterator begin() { return FactIDs.begin(); }
  const_iterator begin() const { return FactIDs.begin(); }

  iterator end() { return FactIDs.end(); }
  const_iterator end() const { return FactIDs.end(); }

  bool isEmpty() const { return FactIDs.size() == 0; }

  // Return true if the set holds no definitely-held positive capability.
  // It may hold negative or try-held facts, unlike isEmpty, which tests
  // the set itself.
  bool holdsNoCapability(FactManager &FactMan) const {
    for (const auto FID : *this) {
      if (!FactMan[FID].negative() && FactMan[FID].definitelyHeld())
        return false;
    }
    return true;
  }

  void addLockByID(FactID ID) { FactIDs.push_back(ID); }

  FactID addLock(FactManager &FM, const FactEntry *Entry) {
    FactID F = FM.newFact(Entry);
    FactIDs.push_back(F);
    return F;
  }

  bool removeLock(FactManager& FM, const CapabilityExpr &CapE) {
    unsigned n = FactIDs.size();
    if (n == 0)
      return false;

    for (unsigned i = 0; i < n-1; ++i) {
      if (FM[FactIDs[i]].matches(CapE)) {
        FactIDs[i] = FactIDs[n-1];
        FactIDs.pop_back();
        return true;
      }
    }
    if (FM[FactIDs[n-1]].matches(CapE)) {
      FactIDs.pop_back();
      return true;
    }
    return false;
  }

  std::optional<FactID> replaceLock(FactManager &FM, iterator It,
                                    const FactEntry *Entry) {
    if (It == end())
      return std::nullopt;
    FactID F = FM.newFact(Entry);
    *It = F;
    return F;
  }

  std::optional<FactID> replaceLock(FactManager &FM, const CapabilityExpr &CapE,
                                    const FactEntry *Entry) {
    return replaceLock(FM, findLockIter(FM, CapE), Entry);
  }

  iterator findLockIter(FactManager &FM, const CapabilityExpr &CapE) {
    return llvm::find_if(*this,
                         [&](FactID ID) { return FM[ID].matches(CapE); });
  }

  const FactEntry *findLock(FactManager &FM, const CapabilityExpr &CapE) const {
    auto I =
        llvm::find_if(*this, [&](FactID ID) { return FM[ID].matches(CapE); });
    return I != end() ? &FM[*I] : nullptr;
  }

  const FactEntry *findLockUniv(FactManager &FM,
                                const CapabilityExpr &CapE) const {
    auto I = llvm::find_if(
        *this, [&](FactID ID) -> bool { return FM[ID].matchesUniv(CapE); });
    return I != end() ? &FM[*I] : nullptr;
  }

  const FactEntry *findPartialMatch(FactManager &FM,
                                    const CapabilityExpr &CapE) const {
    auto I = llvm::find_if(*this, [&](FactID ID) -> bool {
      return FM[ID].partiallyMatches(CapE);
    });
    return I != end() ? &FM[*I] : nullptr;
  }

  bool containsMutexDecl(FactManager &FM, const ValueDecl* Vd) const {
    auto I = llvm::find_if(
        *this, [&](FactID ID) -> bool { return FM[ID].valueDecl() == Vd; });
    return I != end();
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
    // can still be resolved (see getTrylockCallExpr()).
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
  // Variables whose storage is reachable through an escaped reference
  // (address taken, captured or bound by non-const reference): a mutation
  // through the reference is invisible to the map, so a merge of such a
  // variable's definitions must not be resolved (see getTrylockCallExpr()).
  llvm::SmallPtrSet<const NamedDecl *, 4> EscapedDecls;
  // Memoized constant values of canonical definitions, keyed by definition
  // ID (std::nullopt: does not constant-evaluate): intersectContexts()
  // consults the same definitions at every join they reach.
  llvm::DenseMap<unsigned, std::optional<llvm::APSInt>> ConstantValues;
  // Definitions whose chain of prior definitions holds constants only, all
  // the way to the variable's declaration (chainNonConstantDefs()). Only
  // intersectBackEdge() ever changes an existing definition, and it clears
  // this set when it does.
  llvm::DenseSet<unsigned> CleanChains;

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

  void markEscaped(const NamedDecl *D) { EscapedDecls.insert(D); }
  bool isEscaped(const NamedDecl *D) const { return EscapedDecls.count(D); }

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
  /// is what resolution asks about -- getTrylockCallExpr() and phiAbsorbs()
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
    bool Known = walkChain(D, ID, /*StopAt=*/0, [&](unsigned Def) {
      if (CleanChains.contains(Def))
        return ChainVisit::Prune;
      if (!constantValue(Def))
        Defs.insert(Def);
      return ChainVisit::Follow;
    });
    if (Known && Defs.empty())
      if (unsigned Canon = getCanonicalDefinitionID(ID))
        CleanChains.insert(Canon);
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
    auto Covers = [](const llvm::SmallDenseSet<unsigned, 8> &Defs,
                     const llvm::SmallDenseSet<unsigned, 8> &Other) {
      return Defs.size() >= Other.size() &&
             llvm::all_of(Other,
                          [&Defs](unsigned Def) { return Defs.contains(Def); });
    };
    if (Covers(Defs1, Defs2))
      return Canon1;
    if (Covers(Defs2, Defs1))
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
      // was overwritten before this join, which leaves its try-held fact
      // unbalanced for the lockset to diagnose here.
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
  void traverseCFG(CFG *CFGraph, const PostOrderCFGView *SortedGraph,
                   std::vector<CFGBlockInfo> &BlockInfo);

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
  void VisitUnaryOperator(const UnaryOperator *UO);
  void VisitLambdaExpr(const LambdaExpr *LE);
  void VisitCallExpr(const CallExpr *CE);
  void VisitCXXConstructExpr(const CXXConstructExpr *CE);

private:
  void markEscapedIfDeclRef(const Expr *E);
  void markEscapedRefBindings(const InitListExpr *ILE);
};

} // namespace

// The one rule for marking a variable whose storage becomes reachable
// through a reference: it can then be mutated without a visible assignment.
// Shared by every escape site so they cannot drift apart; IgnoreParenCasts,
// because an explicit cast (`(bool &)b`) hides the variable just as well as
// an implicit one.
void VarMapBuilder::markEscapedIfDeclRef(const Expr *E) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenCasts()))
    VMap->markEscaped(DRE->getDecl());
}

// Marks variables bound to non-const reference members in an aggregate
// initialization (`struct W { bool &b; }; W w{ok};`): the aggregate can
// mutate them without a visible assignment, like any reference binding.
void VarMapBuilder::markEscapedRefBindings(const InitListExpr *ILE) {
  const RecordDecl *RD = ILE->getType()->getAsRecordDecl();
  if (!RD || RD->isUnion())
    return; // A union cannot have a reference member.
  auto FI = RD->field_begin(), FE = RD->field_end();
  for (unsigned I = 0, N = ILE->getNumInits(); I < N && FI != FE; ++I, ++FI) {
    const Expr *Init = ILE->getInit(I);
    if (!Init)
      continue;
    QualType FT = FI->getType();
    if (FT->isReferenceType() && !FT.getNonReferenceType().isConstQualified())
      markEscapedIfDeclRef(Init);
    else if (const auto *Nested = dyn_cast<InitListExpr>(Init))
      markEscapedRefBindings(Nested);
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
        modifiedCtx = true;
      } else if (T->isReferenceType() && E &&
                 !T.getNonReferenceType().isConstQualified()) {
        // Binding a non-const reference to a variable lets the variable be
        // mutated without a visible assignment.
        markEscapedIfDeclRef(E);
      }
      // Aggregate initialization can bind non-const reference members.
      if (const auto *ILE =
              dyn_cast_or_null<InitListExpr>(E ? E->IgnoreParenImpCasts()
                                               : nullptr))
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

// Marks a variable whose address is taken: it can then be mutated without a
// visible assignment.
void VarMapBuilder::VisitUnaryOperator(const UnaryOperator *UO) {
  if (UO->getOpcode() != UO_AddrOf)
    return;
  markEscapedIfDeclRef(UO->getSubExpr());
}

// Marks variables captured by reference in a lambda: any later call may
// mutate them without a visible assignment.
void VarMapBuilder::VisitLambdaExpr(const LambdaExpr *LE) {
  for (const LambdaCapture &LC : LE->captures()) {
    if (!LC.capturesVariable() || LC.getCaptureKind() != LCK_ByRef)
      continue;
    const ValueDecl *VD = LC.getCapturedVar();
    VMap->markEscaped(VD);
    // A reference init-capture (`[&x = b]`) binds like a reference
    // declaration: the escaped variable is the one in the initializer.
    if (const auto *IC = dyn_cast<VarDecl>(VD); IC && IC->isInitCapture())
      if (const Expr *Init = IC->getInit())
        markEscapedIfDeclRef(Init);
  }
}

// Invalidates local variable definitions if variable escaped.
void VarMapBuilder::VisitCallExpr(const CallExpr *CE) {
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
    }
  }
  // A back edge is the only thing that ever changes an existing definition
  // (in place, above), which can make a memoized clean chain stale.
  CleanChains.clear();
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

class LockableFactEntry final : public FactEntry {
private:
  /// Reentrancy depth: incremented when a capability has been acquired
  /// again after its initial acquisition -- by a reentrant acquire, or by
  /// a try-acquire over a definite hold.
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

  unsigned int getReentrancyDepth() const override { return ReentrancyDepth; }

  bool definitelyHeld() const override {
    return !tryHeld() || ReentrancyDepth > 0;
  }

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
      FSet.replaceLock(FactMan, entry, RFact);
    } else {
      Handler.handleDoubleLock(entry.getKind(), entry.toString(), loc(),
                               entry.loc(), false);
    }
  }

  void handleUnlock(FactSet &FSet, FactManager &FactMan,
                    const CapabilityExpr &Cp, SourceLocation UnlockLoc,
                    bool FullyRemove,
                    ThreadSafetyHandler &Handler) const override {
    FSet.removeLock(FactMan, Cp);

    if (const FactEntry *RFact = leaveReentrant(FactMan)) {
      // This capability remains reentrantly acquired.
      FSet.addLock(FactMan, RFact);
    } else if (!Cp.negative()) {
      // The release's negative fact supersedes a weak one (not-held on
      // only some paths, see intersectAndWarn()) that may coexist with
      // the released hold; drop it rather than duplicate.
      FSet.removeLock(FactMan, !Cp);
      auto *NegFact = FactMan.createFact<LockableFactEntry>(!Cp, LK_Exclusive,
                                                            UnlockLoc);
      // Releasing a hold that a try-acquire's success proved spends the
      // call's stored result: it stays truthy, but no longer witnesses a
      // live hold (see SpentTryLock).
      if (tryLockCall())
        NegFact->setSpentTryLock(tryLockCall());
      FSet.addLock(FactMan, NegFact);
    }
  }

  // Return an updated FactEntry one level deeper, or nullptr if another
  // acquisition cannot nest in this capability: the kinds must match, and
  // a blocking acquire can only reacquire a reentrant capability. A
  // conditional try-acquire can always nest -- at runtime it may simply
  // fail. This checks only the capability, not the fact's held state:
  // which transitions are permitted is checked by the caller.
  const FactEntry *attemptReenter(FactManager &FactMan, LockKind ReenterKind,
                                  bool Conditional = false) const {
    if (!Conditional && !reentrant())
      return nullptr;
    if (kind() != ReenterKind)
      return nullptr;
    auto *NewFact = FactMan.createFact<LockableFactEntry>(*this);
    NewFact->ReentrancyDepth++;
    return NewFact;
  }

  // Return an updated FactEntry if we are releasing a capability previously
  // acquired reentrant (or conditionally), nullptr otherwise.
  const FactEntry *leaveReentrant(FactManager &FactMan) const {
    if (!ReentrancyDepth)
      return nullptr;
    auto *NewFact = FactMan.createFact<LockableFactEntry>(*this);
    NewFact->ReentrancyDepth--;
    return NewFact;
  }

  static bool classof(const FactEntry *A) {
    return A->getFactEntryKind() == Lockable;
  }
};

/// The location for an unmatched-unlock "released here" note: the negative
/// fact's location if one exists -- unless it came from a try-acquire's
/// failure edge (getEdgeLockset()), which records where the call failed,
/// not a release, and the note would misread it.
static SourceLocation unmatchedUnlockNoteLoc(const FactSet &FSet,
                                             FactManager &FactMan,
                                             const CapabilityExpr &Cp) {
  if (const FactEntry *Neg = FSet.findLock(FactMan, !Cp);
      Neg && !Neg->tryLockCall())
    return Neg->loc();
  return SourceLocation();
}

/// Decide if this unlock unconditionally releases a capability that is only
/// try-held; returns true if the release was handled here.
/// Diagnose like an unmatched unlock and leave the negative fact behind:
/// the thread provably does not hold the capability afterwards, whether the
/// try-acquire succeeded or failed.
/// With a null \p Handler (a scoped guard's destructor, from FullyRemove=true)
/// the fact is kept unchanged: it may record an acquisition the guard does not
/// own, which the destructor's conditional release cannot pair with.
static bool handleUncheckedTryHeldUnlock(FactSet &FSet, FactManager &FactMan,
                                         const FactEntry &Fact,
                                         const CapabilityExpr &Cp,
                                         SourceLocation UnlockLoc,
                                         ThreadSafetyHandler *Handler) {
  if (Fact.definitelyHeld())
    return false;
  if (Handler) {
    Handler->handleUnmatchedUnlock(Cp.getKind(), Cp.toString(), UnlockLoc,
                                   SourceLocation(), true);
    FSet.removeLock(FactMan, Cp);
    // A pre-existing negative fact survives a try-acquire (it is consumed
    // only on the success edge), so do not add a duplicate over it. A
    // weak negative (not-held on only some paths) is superseded: the
    // release proves not-held on every path from here.
    if (!Cp.negative()) {
      const FactEntry *Neg = FSet.findLock(FactMan, !Cp);
      if (Neg && Neg->weak())
        FSet.removeLock(FactMan, !Cp);
      if (!Neg || Neg->weak())
        FSet.addLock(FactMan, FactMan.createFact<LockableFactEntry>(
                                  !Cp, LK_Exclusive, UnlockLoc));
    }
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

  ArrayRef<UnderlyingCapability> getManaged() const {
    return getTrailingObjects(ManagedSize);
  }

public:
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
      const auto *Entry = FSet.findLock(FactMan, UnderlyingMutex.Cap);
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
      FSet.removeLock(FactMan, Cp);
  }

  static bool classof(const FactEntry *A) {
    return A->getFactEntryKind() == ScopedLockable;
  }

private:
  void lock(FactSet &FSet, FactManager &FactMan, const CapabilityExpr &Cp,
            LockKind kind, SourceLocation loc,
            ThreadSafetyHandler *Handler) const {
    if (const auto It = FSet.findLockIter(FactMan, Cp); It != FSet.end()) {
      const auto &Fact = cast<LockableFactEntry>(FactMan[*It]);
      if (const FactEntry *RFact = Fact.attemptReenter(FactMan, kind)) {
        // This capability has been reentrantly acquired.
        FSet.replaceLock(FactMan, It, RFact);
      } else if (Handler) {
        Handler->handleDoubleLock(Cp.getKind(), Cp.toString(), Fact.loc(), loc,
                                  /*MaybeHeld=*/Fact.tryHeld());
      }
    } else {
      FSet.removeLock(FactMan, !Cp);
      FSet.addLock(FactMan, FactMan.createFact<LockableFactEntry>(Cp, kind, loc,
                                                                  Managed));
    }
  }

  void unlock(FactSet &FSet, FactManager &FactMan, const CapabilityExpr &Cp,
              SourceLocation loc, ThreadSafetyHandler *Handler) const {
    // The release's negative fact supersedes a weak one (not-held on only
    // some paths, see intersectAndWarn()) that may coexist with the
    // released hold; drop it rather than duplicate.
    if (const FactEntry *Neg = FSet.findLock(FactMan, !Cp);
        Neg && Neg->weak())
      FSet.removeLock(FactMan, !Cp);
    if (const auto It = FSet.findLockIter(FactMan, Cp); It != FSet.end()) {
      const auto &Fact = cast<LockableFactEntry>(FactMan[*It]);
      if (handleUncheckedTryHeldUnlock(FSet, FactMan, Fact, Cp, loc, Handler))
        return;
      if (const FactEntry *RFact = Fact.leaveReentrant(FactMan)) {
        // This capability remains reentrantly acquired.
        FSet.replaceLock(FactMan, It, RFact);
        return;
      }

      auto *NegFact =
          FactMan.createFact<LockableFactEntry>(!Cp, LK_Exclusive, loc);
      // As in LockableFactEntry::handleUnlock(): releasing a hold proved
      // by a try-acquire's success spends the call's stored result.
      if (Fact.tryLockCall())
        NegFact->setSpentTryLock(Fact.tryLockCall());
      FSet.replaceLock(FactMan, It, NegFact);
    } else if (Handler) {
      Handler->handleUnmatchedUnlock(Cp.getKind(), Cp.toString(), loc,
                                     unmatchedUnlockNoteLoc(FSet, FactMan, Cp),
                                     false);
    }
  }
};

/// Class which implements the core thread safety analysis routines.
class ThreadSafetyAnalyzer {
  friend class BuildLockset;
  friend class threadSafety::BeforeSet;

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
  // orders lose the same fact twice -- e.g. (try-held, no-fact, try-held):
  // the fact-free middle predecessor removes it from the entry set with a
  // diagnostic, then the last predecessor re-supplies it one-sided and
  // would diagnose the same leak again (intersectAndWarn()).
  llvm::StringSet<> NeverCheckedWarned;
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
    /// Whether try-held facts were created for these capabilities at the
    /// call: false for a scoped lockable's construction, whose underlying
    /// capabilities the scoped fact manages instead. getEdgeLockset() only
    /// re-materializes a lost fact for a call that tracked one to lose.
    bool TracksFacts = false;
  };
  // Maps each try-acquire call to its attributes' capabilities, recorded
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
  const FactEntry *cloneWithTryLock(const FactEntry &FE, const Expr *Call,
                                    bool Conditional);
  const FactEntry *cloneAsWeak(const FactEntry &FE);
  void removeLock(FactSet &FSet, const CapabilityExpr &CapE,
                  SourceLocation UnlockLoc, bool FullyRemove, LockKind Kind);

  template <typename AttrType>
  void getMutexIDs(CapExprSet &Mtxs, AttrType *Attr, const Expr *Exp,
                   const NamedDecl *D, til::SExpr *Self = nullptr);

  void recordTryAcquireCall(const Expr *Exp, const NamedDecl *D,
                            til::SExpr *Self = nullptr,
                            TryAcquireCaps *NoExprCaps = nullptr);
  void recordTryAcquireCalls(const PostOrderCFGView *SortedGraph);
  void reconcileTryAcquireCaps(TryAcquireCaps &Caps);

  /// Intermediate state for decodeTrylockBranch.
  struct TrylockDecode {
    /// The try-acquire call reached in the AST walk.
    const CallExpr *TrylockCall = nullptr;
    /// The condition tests the negated call result.
    bool Negate = false;
    /// Set when the branched-on variable merges the call's result with a
    /// constant: the branch-condition truthiness of the edges where the
    /// value may be the constant rather than the call's result (see
    /// decodeTrylockCond()).
    std::optional<bool> AmbiguousCond;
    /// The second of two structurally identical try-acquire calls whose
    /// merged result the condition branches on; null otherwise.
    const CallExpr *MergedCall = nullptr;
  };

  void decodeTrylockCond(const Stmt *Cond, LocalVarContext C, TrylockDecode &D);

  /// How one edge of a terminator's branch resolves one capability of the
  /// branched-on try-acquire call.
  enum class CapResolution : uint8_t {
    Unknown, ///< The edge does not decide this capability's outcome.
    Success, ///< The call acquired the capability.
    Failure, ///< The call did not acquire the capability.
  };

  /// One capability of the call with the resolution one branch direction
  /// or edge proves for it.
  struct TrylockEdgeCap {
    CapabilityExpr Cap;
    LockKind Kind;
    CapResolution Resolution;
  };

  struct TrylockBranch {
    /// The try-acquire call whose result the terminator
    /// branches on, or null if it does not branch on one.
    const CallExpr *TrylockCall = nullptr;
    /// When the branched-on variable merges the results of two structurally
    /// identical try-acquire calls, the second path's call (TrylockCall
    /// resolves to the first path's); null otherwise. A join over facts of
    /// the two calls keeps the resolved origin (intersectAndWarn()).
    const CallExpr *TrylockCall2 = nullptr;
    /// The call may not have executed on edges of this direction (the
    /// branched-on variable merges its result with a constant): each
    /// capability's resolution holds only if it did (TrylockEdge's
    /// Ambiguous).
    bool AmbiguousTrue = false, AmbiguousFalse = false;
    /// The call's capabilities for each branch direction.
    SmallVector<TrylockEdgeCap, 1> OnTrue, OnFalse;
  };

  // Memoize the decodeTrylockBranch result by BlockID.
  llvm::SmallDenseMap<unsigned, TrylockBranch, 8> TerminatorTrylockCache;

  const TrylockBranch &decodeTrylockBranch(const CFGBlock *Block);

  /// The try-acquire calls a block's terminator branches on.
  struct TerminatorTrylockCall {
    const CallExpr *TrylockCall = nullptr;
    const CallExpr *TrylockCall2 = nullptr;
  };
  TerminatorTrylockCall getTerminatorTrylockCall(const CFGBlock *Block);
  const CallExpr *getConditionTrylockCallExpr(const CFGBlock *Block,
                                              bool *ResolvesAllPaths = nullptr,
                                              const CallExpr **MergedCall =
                                                  nullptr);

  /// One edge from a TrylockBranch.
  struct TrylockEdge {
    const CallExpr *TrylockCall = nullptr;
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
                        SourceLocation JoinLoc, LockErrorKind EntryLEK,
                        LockErrorKind ExitLEK,
                        const Expr *RebranchTryLock = nullptr,
                        bool RebranchResolvesAllPaths = true,
                        const Expr *RebranchTryLock2 = nullptr,
                        const llvm::SmallPtrSetImpl<const Expr *>
                            *CheckedAroundLoop = nullptr);

  void intersectAndWarn(FactSet &EntrySet, const FactSet &ExitSet,
                        SourceLocation JoinLoc, LockErrorKind LEK) {
    intersectAndWarn(EntrySet, ExitSet, JoinLoc, LEK, LEK);
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

  checkAcquiredCapability(FSet, *Entry, ReqAttr);

  if (const FactEntry *Cp = FSet.findLock(FactMan, *Entry)) {
    if (Entry->tryHeld()) {
      // Try-acquiring a capability that is already tracked deepens a
      // definite hold, regardless of reentrancy: unlike a blocking
      // acquire, the call cannot deadlock -- it simply fails when the
      // capability cannot be recursively acquired (and even a reentrant
      // capability's success is not guaranteed, e.g. by a recursion
      // limit). The deepened fact becomes conditional at its top level,
      // resolved by a branch on the result like any other try-held fact.
      if (const auto *LCp = dyn_cast<LockableFactEntry>(Cp);
          LCp && !Cp->tryHeld())
        if (const FactEntry *RFact = LCp->attemptReenter(
                FactMan, Entry->kind(), /*Conditional=*/true)) {
          FSet.replaceLock(FactMan, *Cp,
                           cloneWithTryLock(*RFact, Entry->tryLockCall(),
                                            /*Conditional=*/true));
          return;
        }
      // Otherwise the model cannot track this call's acquisition:
      // different kinds or repeat try-held cannot be expressed.
      Handler.handleDoubleLock(Entry->getKind(), Entry->toString(), Cp->loc(),
                               Entry->loc(), /*MaybeHeld=*/Cp->tryHeld());
      return;
    }
    if (Cp->tryHeld()) {
      if (Entry->asserted()) {
        // An assert directly upgrades the lock to being held, without a
        // diagnostic: it claims exactly that knowledge.
        FSet.replaceLock(FactMan, *Entry, Entry);
        return;
      }
      // A reentrant acquire of the same kind deepens the try-held fact.
      if (const auto *LCp = dyn_cast<LockableFactEntry>(Cp))
        if (const FactEntry *RFact =
                LCp->attemptReenter(FactMan, Entry->kind())) {
          FSet.replaceLock(FactMan, *Cp, RFact);
          return;
        }
      // Warn that this TryHeld -> Held transition is invalid.
      assert(!Entry->tryLockCall() &&
             "branch edges resolve facts in place; they do not re-acquire");
      Handler.handleDoubleLock(Entry->getKind(), Entry->toString(), Cp->loc(),
                               Entry->loc(), /*MaybeHeld=*/true);
      // Subsequently, if the program didn't deadlock, it is now asserted
      // locked.
      FSet.replaceLock(FactMan, *Entry, Entry);
      return;
    }
    if (!Entry->asserted())
      Cp->handleLock(FSet, FactMan, *Entry, Handler);
  } else {
    FSet.addLock(FactMan, Entry);
  }
}

/// The checks an acquisition performs: consume (or require) the negative
/// capability, and check acquired_before/acquired_after ordering. A
/// try-acquire attempts the acquisition, so a try-held \p Entry is checked
/// the same way -- once, at the call. The negative capability is consumed
/// either way: after the call the capability is possibly held, so a
/// negative fact that predates it no longer describes the state (and must
/// not later testify that this call failed); on the
/// call's failure edge getEdgeLockset() re-establishes the negative fact,
/// carrying the call as its origin.
void ThreadSafetyAnalyzer::checkAcquiredCapability(FactSet &FSet,
                                                   const FactEntry &Entry,
                                                   bool ReqAttr) {
  if (!ReqAttr && !Entry.negative()) {
    // look for the negative capability, and remove it from the fact set.
    // A weak negative fact (not-held on only some paths, see
    // intersectAndWarn()) is likewise consumed -- after the call the
    // capability is possibly held everywhere -- but does not satisfy the
    // requirement: it proves nothing on the other paths.
    CapabilityExpr NegC = !Entry;
    const FactEntry *Nen = FSet.findLock(FactMan, NegC);
    if (Nen)
      FSet.removeLock(FactMan, NegC);
    if (!Nen || Nen->weak()) {
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

/// Clone \p FE with its try-acquire origin and try-held flag replaced.
const FactEntry *ThreadSafetyAnalyzer::cloneWithTryLock(const FactEntry &FE,
                                                        const Expr *Call,
                                                        bool Conditional) {
  auto *NewFact =
      FactMan.createFact<LockableFactEntry>(cast<LockableFactEntry>(FE));
  NewFact->setTryLock(Call, Conditional);
  return NewFact;
}

/// Clone the negative fact \p FE marked weak: known to hold on only some
/// paths into the current program point.
const FactEntry *ThreadSafetyAnalyzer::cloneAsWeak(const FactEntry &FE) {
  assert(FE.negative() && "only negative facts are tracked as weak");
  auto *NewFact =
      FactMan.createFact<LockableFactEntry>(cast<LockableFactEntry>(FE));
  NewFact->setWeak();
  return NewFact;
}

/// Add a try-held fact for the capability \p CE acquired by the try-acquire
/// call \p Call at \p Loc; the fact remembers its originating call.
void ThreadSafetyAnalyzer::addTryLock(FactSet &FSet, const CapabilityExpr &CE,
                                      LockKind LK, SourceLocation Loc,
                                      const Expr *Call) {
  auto *Fact = FactMan.createFact<LockableFactEntry>(CE, LK, Loc);
  Fact->setTryLock(Call, /*Conditional=*/true);
  addLock(FSet, Fact);
}

/// Remove a lock from the lockset, warning if the lock is not there.
/// \param UnlockLoc The source location of the unlock (only used in error msg)
void ThreadSafetyAnalyzer::removeLock(FactSet &FSet, const CapabilityExpr &Cp,
                                      SourceLocation UnlockLoc,
                                      bool FullyRemove, LockKind ReceivedKind) {
  if (Cp.shouldIgnore())
    return;

  const FactEntry *LDat = FSet.findLock(FactMan, Cp);
  if (!LDat) {
    Handler.handleUnmatchedUnlock(Cp.getKind(), Cp.toString(), UnlockLoc,
                                  unmatchedUnlockNoteLoc(FSet, FactMan, Cp),
                                  false);
    return;
  }

  if (handleUncheckedTryHeldUnlock(FSet, FactMan, *LDat, Cp, UnlockLoc,
                                   &Handler))
    return;

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

// If Cond can be traced back to a try-acquire function call, the `D` variable
// will be populated with the call and with how the branched-on value relates
// to its result -- negation (e.g. `if (!mu.tryLock(...))`), a merge with a
// constant, or a merge of two structurally identical calls.
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
    // The reasoning below assumes every assignment to the variable is
    // visible in the map. A variable whose reference has escaped (captured
    // or bound by reference, address taken) can be mutated by any call in
    // between, so neither its direct definitions nor its merges identify
    // the branched-on value.
    if (LocalVarMap.isEscaped(DRE->getDecl()))
      return;
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
    ASTContext &ACtx = DRE->getDecl()->getASTContext();
    const Expr *NonConst = nullptr, *NonConst2 = nullptr;
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
      // first path's call: its fact is the one in the entry set wherever
      // this merge is branched on, and joins have verified the two paths'
      // states agree.
      llvm::FoldingSetNodeID ID1, ID2;
      NonConst->IgnoreParens()->Profile(ID1, ACtx, /*Canonical=*/true);
      NonConst2->IgnoreParens()->Profile(ID2, ACtx, /*Canonical=*/true);
      if (ID1 != ID2)
        return;
      // (In the unsound retry-without-checking variant
      // `b = mu.TryLock(); while (work()) b = mu.TryLock();` the second
      // call executes while the first result may still be pending; that
      // collision is diagnosed at the second call itself, in addLock().)
      // The second path's call resolves the same way; report it through
      // MergedCall so a join over the two calls' facts can keep the
      // resolved origin (intersectAndWarn()). Refused if either path's
      // resolution nests another two-call merge, or the two disagree on
      // negation or ambiguity: a single companion call cannot represent
      // that, and refusing just keeps the join conservative.
      const TrylockDecode BeforeD = D;
      D.MergedCall = nullptr;
      decodeTrylockCond(NonConst, NonConstCtx, D);
      const CallExpr *First = D.TrylockCall;
      if (First && !D.MergedCall) {
        TrylockDecode D2 = BeforeD;
        D2.MergedCall = nullptr;
        decodeTrylockCond(NonConst2, NonConstCtx2, D2);
        const CallExpr *Second = D2.TrylockCall;
        if (Second && Second != First && !D2.MergedCall &&
            D2.Negate == D.Negate && D2.AmbiguousCond == D.AmbiguousCond) {
          // The stored expressions were compared above, but they may be
          // hops (a copy through another variable) that resolved to calls
          // of their own: the identical-resolution premise holds for the
          // calls themselves, so compare those.
          llvm::FoldingSetNodeID CID1, CID2;
          First->Profile(CID1, ACtx, /*Canonical=*/true);
          Second->Profile(CID2, ACtx, /*Canonical=*/true);
          if (CID1 == CID2)
            D.MergedCall = Second;
        }
      } else {
        // The first path's resolution nests a two-call merge of its own: a
        // single companion call cannot represent that, so refuse it and
        // keep the join conservative.
        D.MergedCall = nullptr;
      }
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
    // On the ambiguous edges the variable's truthiness is K; the
    // condition's is K adjusted by the negations applied so far.
    D.AmbiguousCond = *K != D.Negate;
    return decodeTrylockCond(NonConst, NonConstCtx, D);
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
      if (getStaticBooleanValue(BOP->getRHS(), TCond, *ASTCtx)) {
        if (!TCond)
          D.Negate = !D.Negate;
        return decodeTrylockCond(BOP->getLHS(), C, D);
      }
      TCond = false;
      if (getStaticBooleanValue(BOP->getLHS(), TCond, *ASTCtx)) {
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
    if (getStaticBooleanValue(COP->getTrueExpr(), TCond, *ASTCtx) &&
        getStaticBooleanValue(COP->getFalseExpr(), FCond, *ASTCtx)) {
      if (TCond && !FCond)
        return decodeTrylockCond(COP->getCond(), C, D);
      if (!TCond && FCond) {
        D.Negate = !D.Negate;
        return decodeTrylockCond(COP->getCond(), C, D);
      }
      return;
    }
    // One arm is a constant of truthiness K, the other is not: like the
    // merged variable above, a branch on the value still identifies the
    // non-constant arm -- on an edge where the value's truthiness is !K it
    // can only be that arm's result. Edges matching K are recorded as
    // ambiguous; only one merge can be resolved per condition.
    bool ArmCond;
    const Expr *NonConstArm = nullptr;
    std::optional<bool> K;
    if (getStaticBooleanValue(COP->getTrueExpr(), ArmCond, *ASTCtx)) {
      K = ArmCond;
      NonConstArm = COP->getFalseExpr();
    } else if (getStaticBooleanValue(COP->getFalseExpr(), ArmCond, *ASTCtx)) {
      K = ArmCond;
      NonConstArm = COP->getTrueExpr();
    }
    if (K && !D.AmbiguousCond) {
      D.AmbiguousCond = *K != D.Negate;
      return decodeTrylockCond(NonConstArm, C, D);
    }
  } else if (const auto *SE = dyn_cast<StmtExpr>(Cond)) {
    if (const auto *CS = SE->getSubStmt(); CS && !CS->body_empty()) {
      if (const auto *E = dyn_cast<Expr>(CS->body_back()))
        return decodeTrylockCond(E, C, D);
    }
  }
}

ThreadSafetyAnalyzer::TerminatorTrylockCall
ThreadSafetyAnalyzer::getTerminatorTrylockCall(const CFGBlock *Block) {
  const TrylockBranch &B = decodeTrylockBranch(Block);
  return {B.TrylockCall, B.TrylockCall2};
}

/// Find the try-acquire call whose result the condition starting at
/// \p Block branches on. Unlike getTerminatorTrylockCall(), this looks
/// through short-circuit evaluation: in a compound condition such as
/// `while (i < n && !ok)`, \p Block tests only `i < n` and the branch on the
/// try-acquire result sits in a successor block of the condition.
///
/// With \p ResolvesAllPaths, also reports whether every outgoing path of
/// \p Block reaches a branch on that same call's result: a short-circuit
/// edge escapes its condition without evaluating the rest, but may itself
/// lead to another branch on the result (`if (c && b) ...; else if (b)`),
/// which is verified by walking each escape edge the same way. A caller
/// weakening a definitely-held fact on the strength of the re-branch needs
/// this: on an escaping path that never re-branches, the weakened fact
/// leaks unresolved (intersectAndWarn()).
const CallExpr *
ThreadSafetyAnalyzer::getConditionTrylockCallExpr(const CFGBlock *Block,
                                                  bool *ResolvesAllPaths,
                                                  const CallExpr **MergedCall) {
  // The walk follows the successor edges of logical-operator terminators,
  // which stay within one condition expression, and the fall-through edge
  // of transition blocks (single successor, no terminator) -- e.g. where a
  // branch join meets a loop back edge, one hop before the loop condition
  // that re-branches on the merged variable. A transition block need not be
  // empty: its statements cannot invalidate the decode, which uses the
  // condition block's own ExitContext, and a write to the branched-on
  // variable in it makes the resolution itself refuse (getTrylockCallExpr).
  // A fall-through edge can reach an earlier block (the transition block's
  // successor is the back edge's target), so the visited set keeps the walk
  // finite; it is shared with the escape walks below (any walk that fails
  // ends the search, and the all-paths check below describes how a merge
  // into a visited block resolves).
  llvm::SmallPtrSet<const CFGBlock *, 8> Visited;
  SmallVector<const CFGBlock *, 4> Escapes;
  const CallExpr *WalkMerged = nullptr;
  auto Walk = [&](const CFGBlock *Block) -> const CallExpr * {
    while (Block) {
      if (!Visited.insert(Block).second) {
        TerminatorTrylockCall T = getTerminatorTrylockCall(Block);
        WalkMerged = T.TrylockCall2;
        return T.TrylockCall;
      }
      if (TerminatorTrylockCall T = getTerminatorTrylockCall(Block);
          T.TrylockCall) {
        WalkMerged = T.TrylockCall2;
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
        if (BOP->getOpcode() == BO_LOr)
          ++SI;
        else
          ++EscapeSI;
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

  const CallExpr *Exp = Walk(Block);
  if (MergedCall)
    *MergedCall = Exp ? WalkMerged : nullptr;
  if (ResolvesAllPaths) {
    *ResolvesAllPaths = Exp != nullptr;
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
        *ResolvesAllPaths = false;
        break;
      }
    }
  }
  return Exp;
}

/// Decode a try-acquire attribute's success value. An expression that does
/// not constant-evaluate reads as false.
static bool getTrySuccessValue(ASTContext &Ctx, const Expr *BrE) {
  bool Result;
  return BrE && getStaticBooleanValue(BrE, Result, Ctx) && Result;
}

/// If the terminator of \p Block branches on the result of a call to a
/// function annotated with try_acquire_capability (possibly negated or stored
/// in a local variable), return the capabilities recorded for the call, each
/// with the resolution every branch direction proves for it.
const ThreadSafetyAnalyzer::TrylockBranch &
ThreadSafetyAnalyzer::decodeTrylockBranch(const CFGBlock *Block) {
  const unsigned BlockID = Block->getBlockID();

  if (auto It = TerminatorTrylockCache.find(BlockID);
      It != TerminatorTrylockCache.end())
    return It->second;
  auto CacheMiss = [&]() -> const TrylockBranch & {
    return TerminatorTrylockCache[BlockID] = TrylockBranch{};
  };

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
  Result.TrylockCall2 = D.MergedCall;
  if (D.AmbiguousCond)
    (*D.AmbiguousCond ? Result.AmbiguousTrue : Result.AmbiguousFalse) = true;
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
  // groups) records nothing here: it creates no try-held facts, and a
  // branch on its result proves nothing.
  if (Result.OnTrue.empty() && Result.OnFalse.empty())
    return CacheMiss();
  return TerminatorTrylockCache[BlockID] = std::move(Result);
}

/// What an edge out of a terminator implies about the branched-on value.
enum class EdgeValue {
  False,      ///< The value is zero on this edge.
  True,       ///< The value is nonzero on this edge.
  Unknown,    ///< The edge does not determine the value.
  Infeasible, ///< The edge cannot be taken (e.g. the implicit default of a
              ///< switch that lists every value of a boolean condition).
};

/// Determine the truthiness of a switch condition along the edge to
/// \p CaseBlock.
static EdgeValue getSwitchEdgeValue(ASTContext &Ctx, const SwitchStmt *SW,
                                    const CFGBlock *CaseBlock) {
  // A case label pins the value -- but only a label belonging to this
  // switch: the implicit fall-out successor can itself be a labeled
  // statement, e.g. a case of an enclosing switch that the fall-out edge
  // falls through into, which says nothing about this switch's condition
  // beyond matching none of its cases (the derivation below).
  auto IsOwnCase = [SW](const CaseStmt *CS) {
    for (const SwitchCase *SC = SW->getSwitchCaseList(); SC;
         SC = SC->getNextSwitchCase())
      if (SC == CS)
        return true;
    return false;
  };
  // The value range [Lo, Hi] a case label covers (a single value unless it
  // is a GNU case range).
  auto GetCaseRange = [&Ctx](const CaseStmt *CS) {
    llvm::APSInt Lo = CS->getLHS()->EvaluateKnownConstInt(Ctx);
    llvm::APSInt Hi =
        CS->getRHS() ? CS->getRHS()->EvaluateKnownConstInt(Ctx) : Lo;
    return std::make_pair(Lo, Hi);
  };
  if (const auto *CS = dyn_cast_if_present<CaseStmt>(CaseBlock->getLabel());
      CS && IsOwnCase(CS)) {
    auto [Lo, Hi] = GetCaseRange(CS);
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
  bool ZeroListed = false, OneListed = false;
  for (const SwitchCase *SC = SW->getSwitchCaseList(); SC;
       SC = SC->getNextSwitchCase()) {
    const auto *CS = dyn_cast<CaseStmt>(SC);
    if (!CS)
      continue;
    auto [Lo, Hi] = GetCaseRange(CS);
    ZeroListed |= Lo <= 0 && Hi >= 0;
    OneListed |= Lo <= 1 && Hi >= 1;
  }
  // Not just bool-typed conditions: an int-typed condition provably 0/1
  // (e.g. a comparison in C) derives the same way.
  const bool IsBool = SW->getCond()->isKnownToHaveBooleanValue();
  if (ZeroListed)
    return IsBool && OneListed ? EdgeValue::Infeasible : EdgeValue::True;
  if (IsBool && OneListed)
    return EdgeValue::False;
  return EdgeValue::Unknown;
}

/// Decode what the edge from \p PredBlock to \p CurrBlock proves about
/// conditional capabilities, selected by the truthiness the edge assigns
/// to the branched-on value. An edge that does not determine the value
/// reports no branch at all: the facts stay untouched either way.
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
    CondVal = getSwitchEdgeValue(
        B.TrylockCall->getCalleeDecl()->getASTContext(), SW, CurrBlock);
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
    Edge.TrylockCall = B.TrylockCall;
    Edge.Infeasible = true;
    return Edge;
  }
  if (CondVal == EdgeValue::Unknown)
    return Edge;

  Edge.TrylockCall = B.TrylockCall;
  // If the branched-on variable merges the call's result with a constant,
  // an edge matching the constant's truthiness does not prove the call
  // executed.
  Edge.Ambiguous = CondVal == EdgeValue::True ? B.AmbiguousTrue
                                              : B.AmbiguousFalse;
  const SmallVectorImpl<TrylockEdgeCap> &Dir =
      CondVal == EdgeValue::True ? B.OnTrue : B.OnFalse;
  Edge.Caps.assign(Dir.begin(), Dir.end());
  return Edge;
}

/// Find the lockset that holds on the edge between PredBlock
/// and CurrBlock.  The edge set is the exit set of PredBlock (passed
/// as the ExitSet parameter) plus any trylocks, which are conditionally held.
///
/// Returns true if the edge is infeasible: a fact already promoted to held
/// proves the branched-on try-acquire succeeded on every path into
/// PredBlock, so the failure edge cannot be taken. The caller skips such
/// edges at joins, like unreachable predecessors.
bool ThreadSafetyAnalyzer::getEdgeLockset(FactSet &Result,
                                          const FactSet &ExitSet,
                                          const CFGBlock *PredBlock,
                                          const CFGBlock *CurrBlock) {
  Result = ExitSet;

  TrylockEdge Edge = resolveTrylockEdge(PredBlock, CurrBlock);
  const CallExpr *Exp = Edge.TrylockCall;
  if (!Exp)
    return false;
  if (Edge.Infeasible)
    return true;

  // If the branched-on variable merges the call's result with a constant,
  // an edge matching the constant's truthiness does not prove the call
  // executed. Each fact decides for itself what such an edge still proves:
  // it resolves as a failure edge for a fact whose own attribute reports no
  // success here (even the call executing would mean failure for that
  // capability, and the call not executing means it was never acquired),
  // while a fact whose attribute reports success is left untouched, like an
  // unresolved condition -- attributes carry their own success values, so
  // one call's capabilities can split both ways across the same edge. A
  // negative fact likewise concludes no infeasibility on such an edge: the
  // edge may be taken with the constant's value, the call never executed.
  // (A fact already promoted to held proves the call executed and succeeded
  // on every path into PredBlock -- the branch that promoted it overwrote
  // the constant -- so the promoted-fact infeasibility check below remains
  // correct even on an ambiguous edge.)
  const bool Ambiguous = Edge.Ambiguous;

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
    assert(llvm::any_of(
               Edge.Caps,
               [&](const TrylockEdgeCap &EC) { return FE.matches(EC.Cap); }) &&
           "try-acquire fact matches neither polarity's capabilities");
    return false;
  };

  // This edge resolves every fact originating from this call, each with its
  // own attribute's polarity. A fact still try-held is promoted to held on
  // the branch on which its attribute reports success, removed on the other
  // branch (TryHeld -> Held / NotHeld); it is resolved with the capability
  // recorded at the call, never a re-translation at this edge, which could
  // name a different capability (e.g. through a pointer reassigned since
  // the call).
  //
  // A fact already promoted to held by an earlier branch on the same result
  // proves its attribute reported success on every path into PredBlock: an
  // edge implying the opposite result cannot be taken, so the caller skips
  // it at joins like an unreachable predecessor (but still analyzes a block
  // this leaves without feasible predecessors, see runAnalysis()); on other
  // edges the promoted fact is kept unchanged -- re-resolving is not a new
  // acquisition, so it keeps its reentrancy depth and source, and the
  // acquisition checks do not run again.
  SmallVector<const FactEntry *> ResolvedTryFacts;
  bool Infeasible = false;
  for (const auto &Fact : Result) {
    const FactEntry &FE = FactMan[Fact];
    if (FE.tryLockCall() != Exp)
      continue;
    if (FE.negative()) {
      // A negative fact recorded on the call's failure edge (below): the
      // call provably failed to acquire this fact's capability on every
      // path here, so an edge on which the capability's own attribute
      // reports success cannot be taken; any other edge is simply
      // consistent with it (attributes carry their own success values, so
      // the test is per fact, not per edge). A weak negative proves the
      // failure on only some paths and cannot rule the edge out; nor can
      // one that merged with a spent-result negative (see SpentTryLock),
      // whose paths carry a truthy result; nor can an ambiguous edge be
      // ruled out at all, since it does not prove the call executed.
      if (!Ambiguous && !FE.weak() && !FE.spentTryLock() &&
          FactSucceedsHere(!FE))
        Infeasible = true;
      continue;
    }
    if (FE.tryHeld()) {
      ResolvedTryFacts.push_back(&FE);
      continue;
    }
    if (!FactSucceedsHere(FE))
      Infeasible = true;
  }
  if (Infeasible)
    return true;
  for (const FactEntry *FE : ResolvedTryFacts) {
    if (FactSucceedsHere(*FE)) {
      // An ambiguous edge does not prove the call executed, so it cannot
      // promote the fact; it stays try-held, like an unresolved condition.
      if (Ambiguous)
        continue;
      // The promoted fact keeps its origin: this promotion is proved by the
      // branch, so joins and later branches on the call's result can
      // recognize it (see intersectAndWarn()). The acquisition checks ran
      // at the call (checkAcquiredCapability()); the proved acquisition now
      // consumes the negative capability the call could only require.
      Result.replaceLock(FactMan, *FE,
                         cloneWithTryLock(*FE, FE->tryLockCall(),
                                          /*Conditional=*/false));
      Result.removeLock(FactMan, !*FE);
    } else if (const FactEntry *Shallower =
                   isa<LockableFactEntry>(FE)
                       ? cast<LockableFactEntry>(FE)->leaveReentrant(FactMan)
                       : nullptr) {
      // Failure edge removes try-held fact from reentrant stack.
      Result.replaceLock(FactMan, *FE,
                         cloneWithTryLock(*Shallower, nullptr,
                                          /*Conditional=*/false));
    } else {
      // Failure edge replaces try-held with unheld.
      Result.removeLock(FactMan, *FE);
      // This edge proves the call did not acquire the fact's capability:
      // record that as a negative fact carrying the call as its origin, so
      // a later branch on the same result stays consistent (an edge
      // implying success is infeasible, above). A weak negative (not-held
      // on only some paths) is upgraded: this edge proves it on all.
      const FactEntry *Neg = Result.findLock(FactMan, !*FE);
      if (!Neg || Neg->weak()) {
        auto *NegFact = FactMan.createFact<LockableFactEntry>(
            !*FE, LK_Exclusive, Exp->getExprLoc());
        NegFact->setTryLock(Exp, /*Conditional=*/false);
        if (Neg)
          Result.replaceLock(FactMan, !*FE, NegFact);
        else
          Result.addLock(FactMan, NegFact);
      }
    }
  }

  // Re-materialize a fact of the call that the analysis lost track of --
  // e.g. dropped at a join whose paths a loop separates -- as held on the
  // edge where its attribute reports success: the branch proves the call
  // acquired the capability. Refused when a negative fact for the
  // capability survives from anywhere but this call: it proves the hold
  // was since released, or another call failed to acquire it, on all
  // paths (a real negative) or on some path (a weak one, see
  // intersectAndWarn()) -- either way the stored result is stale there
  // and the hold must not be resurrected. This call's own failure-edge
  // negative kept weak by a join is consistent with the model: this edge
  // excludes the paths it holds on, so resolve over it. (Its real form
  // already proved this edge infeasible above.) An ambiguous edge proves
  // no acquisition either way -- the call may never have executed -- so it
  // re-materializes nothing.
  if (auto MapIt = TryAcquireCapsMap.find(Exp);
      !Ambiguous && MapIt != TryAcquireCapsMap.end() &&
      MapIt->second.TracksFacts) {
    for (const TrylockEdgeCap &EC : Edge.Caps) {
      if (EC.Resolution != CapResolution::Success)
        continue;
      const CapabilityExpr &CE = EC.Cap;
      if (Result.findLock(FactMan, CE))
        continue;
      if (const FactEntry *Neg = Result.findLock(FactMan, !CE)) {
        if (Neg->tryLockCall() != Exp || Neg->spentTryLock())
          continue;
        Result.removeLock(FactMan, !CE);
      }
      auto *Fact = FactMan.createFact<LockableFactEntry>(CE, EC.Kind,
                                                         Exp->getExprLoc());
      Fact->setTryLock(Exp, /*Conditional=*/false);
      Result.addLock(FactMan, Fact);
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
    // Negative capabilities act like locks excluded.
    if (const FactEntry *LDat = FSet.findLock(FactMan, !Cp)) {
      Handler.handleFunExcludesLock(Cp.getKind(), D->getNameAsString(),
                                    (!Cp).toString(), Loc,
                                    !LDat->definitelyHeld());
      return;
    }

    // If this does not refer to a negative capability in the same class,
    // then stop here.
    if (!inCurrentScope(Cp))
      return;

    // Otherwise the negative requirement must be propagated to the caller.
    // A weak negative fact (not-held on only some paths) does not satisfy
    // it.
    if (const FactEntry *Neg = FSet.findLock(FactMan, Cp);
        !Neg || Neg->weak())
      Handler.handleNegativeNotHeld(D, Cp.toString(), Loc);
    return;
  }

  const FactEntry *LDat = FSet.findLockUniv(FactMan, Cp);
  // A try-held capability does not satisfy a requirement: it is only held on
  // the paths where the try-acquire succeeded.
  if (LDat && !LDat->definitelyHeld())
    LDat = nullptr;
  bool NoError = true;
  if (!LDat) {
    // No exact match found.  Look for a partial match.
    LDat = FSet.findPartialMatch(FactMan, Cp);
    if (LDat && !LDat->definitelyHeld())
      LDat = nullptr;
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
    const FactEntry *LDat = FSet.findLockUniv(FactMan, Cp);
    if (LDat && LDat->definitelyHeld() && LDat->isAtLeast(LK_Shared))
      return; // At least one held — read access is safe.
    // FIXME: try findPartialMatch as a fallback to support
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

  if (const FactEntry *LDat = FSet.findLock(FactMan, Cp)) {
    Handler.handleFunExcludesLock(Cp.getKind(), D->getNameAsString(),
                                  Cp.toString(), Loc, !LDat->definitelyHeld());
  }
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
  // or cleanup function): there is no result to branch on, but a
  // reconciled unconditional acquisition still applies.
  ThreadSafetyAnalyzer::TryAcquireCaps NoExprTryCaps;
  bool NoExprTryCapsRecorded = false;

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

      // Try-acquired capabilities were already recorded for CallExprs, so
      // only a constructor or an expression-less call (a destructor or
      // cleanup function) is recorded here, on its first try-acquire
      // attribute, where its object placeholder is available.
      // The conditional locks are added to our lockset below, from the
      // recorded capabilities in TryAcquireCapsMap.
      case attr::TryAcquireCapability: {
        if (Exp ? (!isa<CXXConstructExpr>(Exp) ||
                   Analyzer->TryAcquireCapsMap.contains(Exp))
                : NoExprTryCapsRecorded)
          break;
        NoExprTryCapsRecorded = true;
        auto PostContextForThisScope =
            LVarCtx.switchToContextForScope(DualLocalVarContext::Post);
        Analyzer->recordTryAcquireCall(Exp, D, Self, &NoExprTryCaps);
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

  // Recording reconciled the polarity groups (recordTryAcquireCall); the
  // capabilities the reconciliation moved out of them are acquired
  // regardless of the call's result: diagnose and add them
  // unconditionally. The diagnostic is emitted here in the walk, not at
  // recording, so that unreachable code stays silent as for every other
  // diagnostic.
  ThreadSafetyAnalyzer::TryAcquireCaps *TryCaps = nullptr;
  if (Exp) {
    if (auto It = Analyzer->TryAcquireCapsMap.find(Exp);
        It != Analyzer->TryAcquireCapsMap.end())
      TryCaps = &It->second;
  } else {
    TryCaps = &NoExprTryCaps;
  }
  if (TryCaps) {
    auto AddRegardless = [&](const CapExprSet &Unconditional,
                             CapExprSet &LocksToAdd) {
      for (const auto &M : Unconditional) {
        Analyzer->Handler.handleTryLockRegardlessOfResult(M.getKind(),
                                                          M.toString(), Loc);
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
      const FactEntry *Fact = FSet.findLock(Analyzer->FactMan, Cp);
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
  if (Exp && Scp.shouldIgnore() && TryCaps) {
    TryCaps->TracksFacts = true;
    for (const auto &M : TryCaps->TruthyExclusive)
      Analyzer->addTryLock(FSet, M, LK_Exclusive, Loc, Exp);
    for (const auto &M : TryCaps->FalsyExclusive)
      Analyzer->addTryLock(FSet, M, LK_Exclusive, Loc, Exp);
    for (const auto &M : TryCaps->TruthyShared)
      Analyzer->addTryLock(FSet, M, LK_Shared, Loc, Exp);
    for (const auto &M : TryCaps->FalsyShared)
      Analyzer->addTryLock(FSet, M, LK_Shared, Loc, Exp);
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

/// Given two facts merging on a join point, possibly warn and decide whether to
/// keep or replace.
///
/// \return  false if we should keep \p A, true if we should take \p B.
bool ThreadSafetyAnalyzer::join(const FactEntry &A, const FactEntry &B,
                                SourceLocation JoinLoc,
                                LockErrorKind EntryLEK) {
  // Whether we can replace \p A by \p B.
  const bool CanModify = EntryLEK != LEK_LockedSomeLoopIterations;

  if (A.tryHeld() != B.tryHeld()) {
    // Held joined with try-held: the merged fact must be the weaker
    // try-held one. Under the same-origin re-branch exemption, only a
    // try-held fact is re-resolved at the edges. An unequal reentrancy
    // depth is diagnosed by intersectAndWarn(), which knows whether the
    // exemption otherwise forgives this join silently.
    return CanModify && B.tryHeld();
  }

  const unsigned int ReentrancyDepthA = A.getReentrancyDepth();
  const unsigned int ReentrancyDepthB = B.getReentrancyDepth();

  if (ReentrancyDepthA != ReentrancyDepthB) {
    Handler.handleMutexHeldEndOfScope(B.getKind(), B.toString(), B.loc(),
                                      JoinLoc, EntryLEK,
                                      /*ReentrancyMismatch=*/true);
    // The mismatch is already diagnosed; keep the fact that guarantees
    // more, to minimize follow-on warnings in the same function: compare
    // reentrancy depth, with a conditional (try-held) top level valued at
    // half a level.
    int ScoreA = 2 * (int)ReentrancyDepthA - (A.tryHeld() ? 1 : 0);
    int ScoreB = 2 * (int)ReentrancyDepthB - (B.tryHeld() ? 1 : 0);
    return CanModify && ScoreB > ScoreA;
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
/// \param JoinLoc The location of the join point for error reporting
/// \param EntryLEK The warning if a mutex is missing from \p EntrySet.
/// \param ExitLEK The warning if a mutex is missing from \p ExitSet.
/// \param RebranchTryLock The try-acquire call whose result the joining
/// block's terminator branches on, if any. A held/try-held difference
/// between facts that both originate from that call is not diagnosed as a
/// lost hold: the paths re-diverge at the terminator, so the merged fact is
/// kept try-held (any reentrancy depth is diagnosed but kept) and
/// re-resolved on the outgoing edges by getEdgeLockset(). A difference
/// against a fact not created by that call is diagnosed normally.
/// \param RebranchResolvesAllPaths Whether every outgoing path of the
/// joining block reaches the branch on \p RebranchTryLock's result (false
/// when the branch was found behind a short-circuit, whose other edge
/// escapes unresolved). When false, weakening a definitely-held fact is
/// diagnosed at the join after all -- the exemption's promise of
/// re-resolution does not hold on the escaping paths -- though the fact is
/// still demoted so the paths that do re-branch resolve it.
/// \param RebranchTryLock2 When the branched-on variable merges the
/// results of two structurally identical try-acquire calls (the merge
/// resolves to \p RebranchTryLock, the first path's call), the second
/// path's call. A join of two try-held facts whose origins are exactly
/// these two calls keeps \p RebranchTryLock as the merged origin instead
/// of clearing it: the variable holds either call's result and both
/// resolve the capability identically, so the outgoing edges resolve the
/// merged fact like a single call's.
void ThreadSafetyAnalyzer::intersectAndWarn(
    FactSet &EntrySet, const FactSet &ExitSet, SourceLocation JoinLoc,
    LockErrorKind EntryLEK, LockErrorKind ExitLEK,
    const Expr *RebranchTryLock, bool RebranchResolvesAllPaths,
    const Expr *RebranchTryLock2,
    const llvm::SmallPtrSetImpl<const Expr *> *CheckedAroundLoop) {
  FactSet EntrySetOrig = EntrySet;

  auto IsTrylockRebranched = [RebranchTryLock](const FactEntry &FE) {
    return RebranchTryLock && FE.tryLockCall() == RebranchTryLock;
  };
  // A one-sided fact under the re-branch exemption is carried (demoted to
  // try-held) on the premise that the fact's stored result is falsy on the
  // side missing it: lost to the call's failure edge, or never acquired.
  // A negative fact on the other side that spent the re-branched call's
  // result (a release of a hold its success had proved, see SpentTryLock)
  // refutes that premise: the result stays truthy there while the
  // capability is no longer held, so re-resolving the carried fact would
  // resurrect the dead hold -- e.g. the release in
  // `if (c) { if (ok) mu.Unlock(); }` followed by another `if (ok)`. A
  // weak such negative (on only some of that side's paths) refutes it the
  // same way.
  auto RebranchVetoedByNegative = [&, this](const FactSet &OtherSet,
                                            const FactEntry &FE) {
    const FactEntry *Neg = OtherSet.findLock(FactMan, !FE);
    return Neg && Neg->spentTryLock() == RebranchTryLock;
  };
  auto DemoteToTryHeld = [&, this](const FactEntry &FE,
                                   LockErrorKind LEK) -> const FactEntry * {
    // Replace a definite hold with a conditional hold. A mismatched
    // reentrancy depth is diagnosed here but kept -- after the warning,
    // the deeper fact guards more of the releases downstream than a
    // stripped one would -- and other differences surface downstream once
    // the edges re-resolve the demoted fact.
    if (FE.getReentrancyDepth() != 0)
      Handler.handleMutexHeldEndOfScope(FE.getKind(), FE.toString(), FE.loc(),
                                        JoinLoc, LEK,
                                        /*ReentrancyMismatch=*/true);
    return cloneWithTryLock(FE, FE.tryLockCall(), /*Conditional=*/true);
  };
  // Warn about a fact the intersection removes (or weakens to try-held).
  // However, a capability managed by a scoped object is exempt -- the
  // scoped fact still knows to release it -- except where the scope itself
  // ends or repeats.
  auto WarnRemovedEntryFact = [&](const FactEntry &EntryFact) {
    if (!EntryFact.managed() || ExitLEK == LEK_LockedSomeLoopIterations ||
        ExitLEK == LEK_NotLockedAtEndOfFunction)
      EntryFact.handleRemovalFromIntersection(EntrySetOrig, FactMan, JoinLoc,
                                              ExitLEK, Handler);
  };
  auto WarnRemovedExitFact = [&](const FactEntry &ExitFact) {
    if (!ExitFact.managed() || EntryLEK == LEK_LockedAtEndOfFunction)
      ExitFact.handleRemovalFromIntersection(ExitSet, FactMan, JoinLoc,
                                             EntryLEK, Handler);
  };
  // Likewise for the beta diagnostic that a try-acquire's possible success
  // is carried into the join (or out of the function) unchecked. Emitted
  // once per (join, acquisition, capability): the pairwise intersection of
  // a many-predecessor join can lose the same fact twice, and the leak it
  // reports is one (see NeverCheckedWarned).
  auto WarnNeverChecked = [&](const FactEntry &FE, SourceLocation Loc,
                              bool AtEndOfFunction) {
    if (!Handler.issueBetaWarnings())
      return;
    SmallString<64> Key;
    llvm::raw_svector_ostream(Key)
        << JoinLoc.getRawEncoding() << ':' << Loc.getRawEncoding() << ':'
        << FE.toString();
    if (!NeverCheckedWarned.insert(Key).second)
      return;
    Handler.handleTryAcquireNeverChecked(FE.getKind(), FE.toString(), Loc,
                                         JoinLoc, AtEndOfFunction);
  };

  // Find locks in ExitSet that conflict or are not in EntrySet, and warn.
  for (const auto &Fact : ExitSet) {
    const FactEntry &ExitFact = FactMan[Fact];

    FactSet::iterator EntryIt = EntrySet.findLockIter(FactMan, ExitFact);
    if (EntryIt != EntrySet.end()) {
      const FactEntry &EntryFact = FactMan[*EntryIt];
      if (EntryFact.tryHeld() != ExitFact.tryHeld()) {
        if (!(IsTrylockRebranched(EntryFact) && IsTrylockRebranched(ExitFact) &&
              RebranchResolvesAllPaths)) {
          // The capability is held on one path but only try-held on the other,
          // and the re-branch exemption does not apply: either the terminator
          // does not re-branch on the try-acquire call both facts originate
          // from, or the re-branch sits behind a short-circuit whose other
          // edge escapes without resolving the result. Warn about this as if
          // the try-held path did not hold the capability at all.
          if (ExitFact.tryHeld())
            WarnRemovedEntryFact(EntryFact);
          else
            WarnRemovedExitFact(ExitFact);
        } else {
          if (EntryLEK != LEK_LockedSomeLoopIterations &&
              EntryFact.getReentrancyDepth() != ExitFact.getReentrancyDepth())
            Handler.handleMutexHeldEndOfScope(ExitFact.getKind(),
                                              ExitFact.toString(),
                                              ExitFact.loc(), JoinLoc, EntryLEK,
                                              /*ReentrancyMismatch=*/true);
        }
      }
      const Expr *EntryOrigin = EntryFact.tryLockCall();
      const bool EntryTryHeld = EntryFact.tryHeld();
      const bool EitherWeak = EntryFact.weak() || ExitFact.weak();
      const Expr *EitherSpent = EntryFact.spentTryLock()
                                    ? EntryFact.spentTryLock()
                                    : ExitFact.spentTryLock();
      if (join(EntryFact, ExitFact, JoinLoc, EntryLEK))
        *EntryIt = Fact;
      // If the two paths hold the capability via different origins, the
      // merged fact is not determined by either try-acquire's result. When
      // both sides were try-held, clearing the origin makes the state
      // permanently unresolvable -- neither call's result can be checked any
      // more -- so diagnose each discarded origin immediately at branch
      // joins (the mixed held/try-held case was already diagnosed above, and
      // loop joins are exempt as elsewhere).
      if (const FactEntry &Merged = FactMan[*EntryIt];
          EntryLEK == LEK_LockedSomePredecessors && Merged.tryLockCall() &&
          EntryOrigin != ExitFact.tryLockCall()) {
        const Expr *ExitOrigin = ExitFact.tryLockCall();
        if (RebranchTryLock2 && EntryTryHeld && ExitFact.tryHeld() &&
            ((EntryOrigin == RebranchTryLock &&
              ExitOrigin == RebranchTryLock2) ||
             (EntryOrigin == RebranchTryLock2 &&
              ExitOrigin == RebranchTryLock))) {
          // ... except when the two origins are exactly the two
          // structurally identical calls whose merged result the joining
          // block's terminator branches on: the branched-on variable holds
          // either call's result and both resolve the capability
          // identically, so the merged fact keeps the resolved call (the
          // first path's, which the merge resolves to) as its origin and
          // the outgoing edges resolve it like a single call's fact.
          if (Merged.tryLockCall() != RebranchTryLock)
            EntrySet.replaceLock(FactMan, EntryIt,
                                 cloneWithTryLock(Merged, RebranchTryLock,
                                                  /*Conditional=*/true));
        } else {
          if (EntryTryHeld && ExitFact.tryHeld()) {
            if (EntryOrigin)
              WarnNeverChecked(Merged, EntryFact.loc(),
                               /*AtEndOfFunction=*/false);
            if (ExitOrigin)
              WarnNeverChecked(Merged, ExitFact.loc(),
                               /*AtEndOfFunction=*/false);
          }
          EntrySet.replaceLock(
              FactMan, EntryIt,
              cloneWithTryLock(Merged, nullptr, Merged.tryHeld()));
        }
      }
      // A negative fact weak or spent on either side is weak or spent in
      // the merged set: proven, or spending a result, on some of that
      // side's paths.
      if (const FactEntry &Merged = FactMan[*EntryIt];
          EntryLEK == LEK_LockedSomePredecessors && Merged.negative() &&
          ((EitherWeak && !Merged.weak()) ||
           (EitherSpent && !Merged.spentTryLock()))) {
        auto *NewFact = FactMan.createFact<LockableFactEntry>(
            cast<LockableFactEntry>(Merged));
        if (EitherWeak)
          NewFact->setWeak();
        if (EitherSpent && !NewFact->spentTryLock())
          NewFact->setSpentTryLock(EitherSpent);
        EntrySet.replaceLock(FactMan, EntryIt, NewFact);
      }
    } else if (ExitFact.negative()) {
      // A negative fact on this predecessor only: keep it in the merged
      // set as a weak fact at branch joins -- evidence for the try-held
      // machinery that the capability was released, or a try-acquire of it
      // failed, on some path (see RebranchVetoedByNegative above and
      // getEdgeLockset()'s re-materialization veto). Under a re-branch
      // this loses nothing: the failure edge re-derives the real negative
      // over it. Skipped in functions without try-lock facts: nothing
      // consults weak facts there.
      if (EntryLEK == LEK_LockedSomePredecessors && !TryAcquireCapsMap.empty()) {
        if (ExitFact.weak())
          EntrySet.addLockByID(Fact);
        else
          EntrySet.addLock(FactMan, cloneAsWeak(ExitFact));
      }
    } else if (IsTrylockRebranched(ExitFact) &&
               !RebranchVetoedByNegative(EntrySetOrig, ExitFact)) {
      // Held on this predecessor only, but the terminator re-branches on
      // the try-acquire that created the fact: demote it to try-held
      // without warning, as getEdgeLockset will re-resolve it on the
      // outgoing edges.
      if (EntryLEK != LEK_LockedSomeLoopIterations) {
        // A re-branch behind a short-circuit does not resolve the result on
        // the escaping edge: a definite hold weakened here can leak there,
        // so it is diagnosed at this join after all (the demotion stands,
        // for the paths that do re-branch).
        if (!RebranchResolvesAllPaths && !ExitFact.tryHeld())
          WarnRemovedExitFact(ExitFact);
        EntrySet.addLock(FactMan, DemoteToTryHeld(ExitFact, EntryLEK));
      }
    } else if (ExitFact.tryHeld()) {
      // The analysis loses track of the try-held fact here -- this
      // predecessor carries a try-acquire result into the join unchecked (or
      // to the end of the function): the capability may be leaked. At a
      // loop join, only when the result is not branched on anywhere inside
      // the loop (CheckedAroundLoop): then the next iteration re-executes
      // the call (or the loop discards the result) while this iteration's
      // possible success was never checked -- a check after the loop sees
      // only the last result and cannot make this sound. Joins without that
      // information (continue joins, see runAnalysis()) stay exempt.
      const bool UncheckedAroundLoop =
          CheckedAroundLoop && ExitFact.tryLockCall() &&
          !CheckedAroundLoop->count(ExitFact.tryLockCall());
      if (EntryLEK != LEK_LockedSomeLoopIterations || UncheckedAroundLoop)
        WarnNeverChecked(ExitFact, ExitFact.loc(),
                         /*AtEndOfFunction=*/EntryLEK ==
                             LEK_LockedAtEndOfFunction);
    } else {
      WarnRemovedExitFact(ExitFact);
    }
  }

  // Find locks in EntrySet that are not in ExitSet, and remove them.
  for (const auto &Fact : EntrySetOrig) {
    const FactEntry *EntryFact = &FactMan[Fact];
    const FactEntry *ExitFact = ExitSet.findLock(FactMan, *EntryFact);

    if (!ExitFact) {
      if (EntryFact->negative()) {
        // As above: a one-sided negative is kept in the merged set as a
        // weak fact at branch joins (or dropped, in functions without
        // try-lock facts); other joins leave the entry set unmodified.
        if (ExitLEK == LEK_LockedSomePredecessors && !EntryFact->weak()) {
          if (!TryAcquireCapsMap.empty())
            EntrySet.replaceLock(FactMan, *EntryFact,
                                 cloneAsWeak(*EntryFact));
          else
            EntrySet.removeLock(FactMan, *EntryFact);
        }
        continue;
      }
      if (IsTrylockRebranched(*EntryFact) &&
          !RebranchVetoedByNegative(ExitSet, *EntryFact)) {
        // As above, but here the fact is kept in the intersection in its
        // demoted try-held form (except at a loop join, where the entry set
        // is left unmodified).
        if (EntryLEK != LEK_LockedSomeLoopIterations &&
            !EntryFact->tryHeld()) {
          // As above: an escaping short-circuit edge means the weakened
          // definite hold is diagnosed at the join after all.
          if (!RebranchResolvesAllPaths)
            WarnRemovedEntryFact(*EntryFact);
          EntrySet.replaceLock(FactMan, *EntryFact,
                               DemoteToTryHeld(*EntryFact, ExitLEK));
        }
        continue;
      }
      if (EntryFact->tryHeld()) {
        // As above, with the unchecked try-acquire on an earlier predecessor.
        // Only at branch joins: a try-held fact missing from a loop's back
        // edge was checked inside the loop, which is not a leak.
        if (ExitLEK == LEK_LockedSomePredecessors)
          WarnNeverChecked(*EntryFact, EntryFact->loc(),
                           /*AtEndOfFunction=*/false);
      } else {
        WarnRemovedEntryFact(*EntryFact);
      }
      if (ExitLEK == LEK_LockedSomePredecessors)
        EntrySet.removeLock(FactMan, *EntryFact);
    }
  }
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
/// remaining capability recorded under exactly one polarity and kind.
/// Exclusive under both polarities stays exclusive. A cross-kind pairing
/// (e.g. exclusive on success, shared on failure) may be a deliberate
/// API, but a single fact cannot represent a hold whose kind varies with
/// the result, so it keeps only the guarantee that holds either way: an
/// unconditional shared hold. handleCall() adds the unconditional groups
/// to the lockset, with the diagnostic.
void ThreadSafetyAnalyzer::reconcileTryAcquireCaps(TryAcquireCaps &Caps) {
  CapExprSet Regardless;
  auto CollectAcquiredOnFailureToo = [&](const CapExprSet &Truthy) {
    for (const auto &M : Truthy)
      if (Caps.FalsyExclusive.contains(M) || Caps.FalsyShared.contains(M))
        Regardless.push_back_nodup(M);
  };
  CollectAcquiredOnFailureToo(Caps.TruthyExclusive);
  CollectAcquiredOnFailureToo(Caps.TruthyShared);
  if (Regardless.empty())
    return;
  for (const auto &M : Regardless)
    (Caps.TruthyExclusive.contains(M) && Caps.FalsyExclusive.contains(M)
         ? Caps.UnconditionalExclusive
         : Caps.UnconditionalShared)
        .push_back_nodup(M);
  auto DropRegardless = [&](CapExprSet &Set) {
    llvm::erase_if(
        Set, [&](const CapabilityExpr &M) { return Regardless.contains(M); });
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
void ThreadSafetyAnalyzer::recordTryAcquireCall(const Expr *Exp,
                                                const NamedDecl *D,
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
}

/// Populate TryAcquireCapsMap for every try-acquire CallExpr in the
/// function, before the lockset walk: a branch on a stored result can
/// precede the call in block order (a loop-top check `if (ok)` above
/// `ok = mu.TryLock()`), and the terminator decode (decodeTrylockBranch)
/// folds the record into its memoized per-capability resolutions. The
/// variable map's per-statement contexts are complete by now
/// (from traverseCFG), so each call's attributes translate in the call's own
/// post-context by replaying the saved contexts block by block.
/// Constructors are excluded: they record in handleCall, where the
/// constructed-object placeholder is available.
void ThreadSafetyAnalyzer::recordTryAcquireCalls(
    const PostOrderCFGView *SortedGraph) {
  for (const CFGBlock *B : *SortedGraph) {
    const CFGBlockInfo &Info = BlockInfo[B->getBlockID()];
    unsigned CtxIndex = Info.EntryIndex;
    LocalVariableMap::Context Ctx = Info.EntryContext;
    for (const auto &BI : *B) {
      std::optional<CFGStmt> CS = BI.getAs<CFGStmt>();
      if (!CS)
        continue;
      const Stmt *S = CS->getStmt();
      // Advance to the post-context of S; a no-op for statements the
      // variable map saved no context for.
      Ctx = LocalVarMap.getNextContext(CtxIndex, S, Ctx);
      const auto *CE = dyn_cast<CallExpr>(S);
      if (!CE)
        continue;
      const auto *D = dyn_cast_or_null<NamedDecl>(CE->getCalleeDecl());
      if (!D || !D->hasAttr<TryAcquireCapabilityAttr>())
        continue;
      // Mirror BuildLockset's post-context attribute translation.
      if (Handler.issueBetaWarnings())
        SxBuilder.setLookupLocalVarExpr(
            [Ctx, this](const NamedDecl *VD) mutable -> const Expr * {
              return LocalVarMap.lookupExpr(VD, Ctx);
            });
      recordTryAcquireCall(CE, D);
    }
  }
  if (Handler.issueBetaWarnings())
    SxBuilder.setLookupLocalVarExpr(nullptr);
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
  recordTryAcquireCalls(SortedGraph);

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
    ExpectedFunctionExitSet.removeLock(FactMan, Lock);

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
    // The try-acquire call whose result the condition starting at this
    // block branches on, if any. Computed lazily on the first join where a
    // set carries a try-acquire fact at all. Each incoming set is scanned
    // once as it arrives (JoinHasTryLockFact accumulates); the entry set
    // itself never gains try-acquire facts from anywhere else.
    const CallExpr *RebranchTryLock = nullptr;
    const CallExpr *RebranchTryLock2 = nullptr;
    bool RebranchTryLockComputed = false;
    bool RebranchResolvesAllPaths = true;
    bool JoinHasTryLockFact = false;
    auto HasTryLockFact = [this](const FactSet &FS) {
      // TryAcquireCapsMap is empty in functions without try-acquires (the
      // common case): skip scanning the fact sets entirely.
      return !TryAcquireCapsMap.empty() && llvm::any_of(FS, [this](FactID ID) {
        return FactMan[ID].tryLockCall();
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
        // The edge cannot be taken (a promoted fact proves the branched-on
        // try-acquire succeeded), or the predecessor itself was analyzed
        // only for coverage and its exit set is dead state either way: skip
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
                           LEK_LockedSomeLoopIterations,
                           LEK_LockedSomeLoopIterations, nullptr);
        } else {
          // Branch join: a difference in the facts created by a try-acquire
          // is demoted to try-held and re-resolved on the outgoing edges if
          // the condition branches on that call's result -- possibly behind
          // short-circuit blocks of a compound condition like `c && ok`.
          if (!RebranchTryLockComputed && !JoinHasTryLockFact)
            JoinHasTryLockFact = HasTryLockFact(PrevLockset);
          if (!RebranchTryLockComputed && JoinHasTryLockFact) {
            // Compute once; the result depends only on CurrBlock, not on
            // *PI. Skipped entirely (the common case) until some fact at
            // this join originates from a try-acquire.
            RebranchTryLock = getConditionTrylockCallExpr(
                CurrBlock, &RebranchResolvesAllPaths, &RebranchTryLock2);
            RebranchTryLockComputed = true;
          }
          intersectAndWarn(CurrBlockInfo->EntrySet, PrevLockset,
                           CurrBlockInfo->EntryLoc, LEK_LockedSomePredecessors,
                           LEK_LockedSomePredecessors, RebranchTryLock,
                           RebranchResolvesAllPaths, RebranchTryLock2);
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
      // A back-edge difference in the facts created by a try-acquire is
      // forgiven when the loop condition branches on that call's result
      // (e.g. a spin loop storing the result), possibly behind
      // short-circuit blocks of a compound condition: the condition's
      // outgoing edges re-resolve the fact, so it does not leak around the
      // loop. Skipped entirely while the function has no try-acquire facts.
      const Expr *RebranchTryLock =
          !TryAcquireCapsMap.empty() ? getConditionTrylockCallExpr(FirstLoopBlock)
                          : nullptr;
      // For the unchecked-result warning: the try-acquire results branched
      // on inside this back edge's natural loop are (or will be, on the
      // next iteration) checked around the loop. Results checked only
      // outside the loop are not: the loop re-executes the call (or
      // discards the result) unchecked.
      llvm::SmallPtrSet<const Expr *, 4> CheckedInLoop;
      const llvm::SmallPtrSetImpl<const Expr *> *CheckedInLoopPtr = nullptr;
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
          TerminatorTrylockCall Checked = getTerminatorTrylockCall(B);
          if (Checked.TrylockCall)
            CheckedInLoop.insert(Checked.TrylockCall);
          // A branch on a merge of two identical calls checks both results.
          if (Checked.TrylockCall2)
            CheckedInLoop.insert(Checked.TrylockCall2);
        }
        CheckedInLoopPtr = &CheckedInLoop;
      }
      // At a loop join the entry set keeps the (weaker) pre-loop facts and
      // the loop condition re-resolves the result each iteration, so the
      // exemption stands even behind a short-circuit.
      intersectAndWarn(PreLoop->EntrySet, LoopEnd->ExitSet, PreLoop->EntryLoc,
                       LEK_LockedSomeLoopIterations,
                       LEK_LockedSomeLoopIterations, RebranchTryLock,
                       /*RebranchResolvesAllPaths=*/true,
                       /*RebranchTryLock2=*/nullptr, CheckedInLoopPtr);
      // A negative fact reaching the loop head on its back edge is
      // evidence that an iteration may have released the capability (or
      // failed to re-acquire it). The head was analyzed before its back
      // edges were seen, so record the negative as a weak fact directly in
      // the head's exit set: the loop's exit edges are processed after
      // this (the sorted graph orders loop bodies before loop successors)
      // and consult it to refuse re-materializing a hold the loop may
      // have released (getEdgeLockset()).
      if (!TryAcquireCapsMap.empty()) {
        for (const auto &Fact : LoopEnd->ExitSet) {
          const FactEntry &FE = FactMan[Fact];
          if (!FE.negative() || PreLoop->ExitSet.findLock(FactMan, FE) ||
              PreLoop->ExitSet.findLock(FactMan, !FE))
            continue;
          if (FE.weak())
            PreLoop->ExitSet.addLockByID(Fact);
          else
            PreLoop->ExitSet.addLock(FactMan, cloneAsWeak(FE));
        }
      }
    }
  }

  // Skip the final check if the exit block is unreachable, or reachable
  // only through infeasible edges: its exit set is dead state (the
  // coverage diagnostics inside the dead blocks have already run).
  if (!Final.Reachable || Final.CoverageOnly)
    return;

  // FIXME: Should we call this function for all blocks which exit the function?
  intersectAndWarn(ExpectedFunctionExitSet, Final.ExitSet, Final.ExitLoc,
                   LEK_LockedAtEndOfFunction, LEK_NotLockedAtEndOfFunction);

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
