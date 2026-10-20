//===- NullabilitySafety.cpp - Nullability safety analysis ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a CFG-based forward dataflow analysis that detects
// dereferences of nullable pointers, tracking nullability narrowing through
// control flow (null checks, early returns, assertions, etc.).
//
// Design overview
//
// The lattice is NullState. Its proof sets, NarrowedVars and NarrowedMembers,
// hold pointers shown non-null and intersect at joins: a pointer stays
// narrowed only if every incoming path agrees. Its taint sets, NullableVars
// and NullableMembers, hold pointers the flow has shown may be null (assigned
// nullptr, reset, moved-from) and union at joins. The guard and alias maps
// (BoolGuards, Aliases, MemberAliases, AddrOfTargets) intersect with value
// equality, so a fact survives a join only when both sides recorded exactly
// the same one.
//
// States are kept per CFG edge, keyed by (predecessor, successor) block ids,
// and a block's entry state is the join of its incoming edge states. This is
// what lets the two branches of a condition receive different states: the
// edge that proves p non-null carries p narrowed, the other does not.
//
// analyzeCondition turns a branch condition into ConditionResult facts of the
// form "pointer X is non-null when the condition is true (or false, when
// Negated)". narrowOnTerminator runs it on a block's terminator and applies
// each fact to the true-edge or false-edge state as its Negated flag says.
// The same machinery serves guard flags (bool ok = p != nullptr; if (ok)),
// __builtin_assume, and ternary arms.
//
// Calls deliberately do not invalidate narrowing, even when a pointer's
// address escapes: doing so floods common output-parameter idioms with false
// positives. Clang's ThreadSafety analysis makes the same trade-off.
//
// The worklist re-processes a block only when its entry state changes. The
// lattice is not provably monotone (facts are both inserted and erased), so
// runNullabilitySafetyAnalysis caps total block visits as a termination safety
// net and suppresses summary inference if the cap fires.
//
// The fixpoint is silent. Diagnostics, evidence, and the return summary come
// from a second pass that visits each reached block once with its converged
// entry state (the UninitializedValues pattern).
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/NullabilitySafety.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Analysis/FlowSensitive/DataflowWorklist.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <optional>
#include <utility>

#define DEBUG_TYPE "nullability-safety"

STATISTIC(NumFunctionsAnalyzed, "Number of functions analyzed");
STATISTIC(NumBlocksProcessed, "Number of CFG blocks processed");
STATISTIC(NumFixpointBailouts,
          "Number of analyses stopped by the block-visit safety cap");
STATISTIC(NumDereferenceWarnings, "Number of nullable dereference warnings");
STATISTIC(NumArithmeticWarnings, "Number of nullable arithmetic warnings");
STATISTIC(NumReturnWarnings, "Number of nullable return warnings");
STATISTIC(NumAssignmentWarnings, "Number of nullable assignment warnings");
STATISTIC(NumArgumentWarnings, "Number of nullable argument warnings");

using namespace clang;

NullabilitySafetyHandler::~NullabilitySafetyHandler() = default;
NullabilitySafetySummaries::~NullabilitySafetySummaries() = default;

//===----------------------------------------------------------------------===//
// Access paths and AST helpers
//===----------------------------------------------------------------------===//

namespace {

/// A chain of field accesses from a root variable, such as var.field,
/// var->inner->field, or this->a.b. Root is nullptr when the path starts at
/// this.
struct MemberAccessPath {
  const VarDecl *Root = nullptr;
  llvm::SmallVector<const FieldDecl *, 2> Fields;

  const FieldDecl *leafField() const {
    assert(!Fields.empty() && "leafField() on empty path");
    return Fields.back();
  }

  bool operator==(const MemberAccessPath &O) const {
    return Root == O.Root && Fields == O.Fields;
  }
  // DenseMap::operator== on MemberAliases compares values with !=.
  bool operator!=(const MemberAccessPath &O) const { return !(*this == O); }
};

} // end anonymous namespace

// DenseMapInfo needs only a hash and an equality predicate; DenseMap has no
// sentinel keys.
template <> struct llvm::DenseMapInfo<MemberAccessPath> {
  static unsigned getHashValue(const MemberAccessPath &P) {
    unsigned H = DenseMapInfo<const VarDecl *>::getHashValue(P.Root);
    for (const auto *FD : P.Fields)
      H = llvm::hash_combine(H,
                             DenseMapInfo<const FieldDecl *>::getHashValue(FD));
    return H;
  }
  static bool isEqual(const MemberAccessPath &L, const MemberAccessPath &R) {
    return L == R;
  }
};

static bool isSmartPointerObject(const Expr *Obj);
static const VarDecl *resolveSmartPtrReference(const VarDecl *VD);

/// Walk a MemberExpr chain to its root, collecting FieldDecls along the way.
/// A std smart pointer's operator-> or operator* continues the chain at the
/// smart pointer, as -> does at a raw pointer, so sp->child and (*sp).child
/// are paths rooted at sp. Returns nullopt if the root is not a VarDecl (via
/// DeclRefExpr) or CXXThisExpr. Root is nullptr for this-> access paths.
static std::optional<MemberAccessPath> decomposeMemberAccess(const Expr *E) {
  llvm::SmallVector<const FieldDecl *, 2> Fields;
  E = E->IgnoreParenImpCasts();

  while (true) {
    if (const auto *ME = dyn_cast<MemberExpr>(E)) {
      if (const auto *FD = dyn_cast<FieldDecl>(ME->getMemberDecl()))
        Fields.push_back(FD);
      else
        return std::nullopt;
      E = ME->getBase()->IgnoreParenImpCasts();
      continue;
    }
    const auto *OCE = dyn_cast<CXXOperatorCallExpr>(E);
    if (!Fields.empty() && OCE && OCE->getNumArgs() == 1 &&
        (OCE->getOperator() == OO_Arrow || OCE->getOperator() == OO_Star) &&
        isSmartPointerObject(OCE->getArg(0))) {
      E = OCE->getArg(0)->IgnoreParenImpCasts();
      continue;
    }
    break;
  }

  if (Fields.empty())
    return std::nullopt;

  std::reverse(Fields.begin(), Fields.end());

  MemberAccessPath Path;
  Path.Fields = std::move(Fields);

  if (isa<CXXThisExpr>(E)) {
    Path.Root = nullptr;
  } else if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    // A local smart pointer reference roots its paths at its referent, as
    // PtrRef::fromExpr does for the reference itself.
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      Path.Root = resolveSmartPtrReference(VD);
    else
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  return Path;
}

/// Whether Path starts with Prefix: same root, and Path's fields begin with
/// Prefix's.
static bool pathHasPrefix(const MemberAccessPath &Path,
                          const MemberAccessPath &Prefix) {
  return Path.Root == Prefix.Root &&
         Path.Fields.size() >= Prefix.Fields.size() &&
         std::equal(Prefix.Fields.begin(), Prefix.Fields.end(),
                    Path.Fields.begin());
}

//===----------------------------------------------------------------------===//
// Lattice
//===----------------------------------------------------------------------===//

static bool isSmartPointerType(QualType Ty);

/// The variable a local smart pointer reference is bound to (auto &a = p or
/// auto &&a = std::move(p)), followed through references to references, or VD
/// itself. A reference is the same object as its referent, so resolving it
/// lets every fact reach both names. Only variables are followed: a member
/// path could name a different object once its root is reassigned.
static const VarDecl *resolveSmartPtrReference(const VarDecl *VD) {
  for (unsigned Depth = 0; Depth < 8; ++Depth) {
    if (!VD->getType()->isReferenceType() || isa<ParmVarDecl>(VD) ||
        !VD->hasLocalStorage() ||
        !isSmartPointerType(VD->getType().getNonReferenceType()))
      return VD;
    const Expr *Init = VD->getInit();
    if (!Init)
      return VD;
    Init = Init->IgnoreParenImpCasts();
    if (const auto *CE = dyn_cast<CallExpr>(Init))
      if (CE->isCallToStdMove() && CE->getNumArgs() == 1)
        Init = CE->getArg(0)->IgnoreParenImpCasts();
    const auto *DRE = dyn_cast<DeclRefExpr>(Init);
    const auto *Target = DRE ? dyn_cast<VarDecl>(DRE->getDecl()) : nullptr;
    if (!Target || Target == VD)
      return VD;
    VD = Target;
  }
  return VD;
}

namespace {

/// A tracked pointer: a local variable or parameter, or a member access path
/// (exactly one of VD and Path is set).
struct PtrRef {
  const VarDecl *VD = nullptr;
  std::optional<MemberAccessPath> Path;

  /// Resolve E (parens and implicit casts ignored) to a tracked pointer, or
  /// nullopt when it is neither a DeclRefExpr to a VarDecl nor a member
  /// access chain rooted at a VarDecl or this. A local smart pointer
  /// reference resolves to its referent.
  static std::optional<PtrRef> fromExpr(const Expr *E) {
    E = E->IgnoreParenImpCasts();
    if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
        return PtrRef{resolveSmartPtrReference(VD), std::nullopt};
      return std::nullopt;
    }
    if (auto P = decomposeMemberAccess(E))
      return PtrRef{nullptr, std::move(P)};
    return std::nullopt;
  }
  /// The tracked variable VD itself (no reference resolution).
  static PtrRef var(const VarDecl *VD) { return PtrRef{VD, std::nullopt}; }
  /// Declared type of the variable or the leaf field.
  QualType getType() const {
    return VD ? VD->getType() : Path->leafField()->getType();
  }
  bool operator==(const PtrRef &O) const {
    return VD == O.VD && Path == O.Path;
  }
};

} // end anonymous namespace

template <> struct llvm::DenseMapInfo<PtrRef> {
  static unsigned getHashValue(const PtrRef &R) {
    return R.VD ? DenseMapInfo<const VarDecl *>::getHashValue(R.VD)
                : DenseMapInfo<MemberAccessPath>::getHashValue(*R.Path);
  }
  static bool isEqual(const PtrRef &L, const PtrRef &R) { return L == R; }
};

namespace {

/// One fact extracted from a branch condition (or a guard variable's
/// initializer): the pointer Ref is non-null when the condition is true
/// (Negated=false) or when it is false (Negated=true).
struct ConditionResult {
  PtrRef Ref;
  bool Negated = false;

  bool operator==(const ConditionResult &O) const {
    return Ref == O.Ref && Negated == O.Negated;
  }
};

/// Per-block dataflow lattice tracking which pointers are narrowed (known
/// non-null) or nullable.
struct NullState {
  /// Pointers proven non-null by control flow (null checks, nonnull init,
  /// etc.). A variable should not be in both NarrowedVars and NullableVars:
  /// narrowing is always erased before re-evaluating nullability on
  /// reassignment.
  llvm::DenseSet<const VarDecl *> NarrowedVars;
  /// Member paths proven non-null: this->field, var.field, and nested paths
  /// like var.inner.field.
  llvm::DenseSet<MemberAccessPath> NarrowedMembers;
  llvm::DenseSet<const VarDecl *> NullableVars;
  /// Member access paths known to be nullable at runtime (e.g., smart pointer
  /// members after reset() or std::move()). Parallels NarrowedMembers.
  llvm::DenseSet<MemberAccessPath> NullableMembers;
  llvm::DenseSet<const VarDecl *> MustNullableVars;
  llvm::DenseSet<MemberAccessPath> MustNullableMembers;

  /// Maps integer-typed guards (bool or int flags), held in a variable or in
  /// a field reached by a member path (d.ok, this->ok), to the null-check
  /// facts they capture, e.g. bool valid = (p != nullptr) stores
  /// {valid -> [(p, Negated=false)]}. Each fact reads exactly like a branch
  /// condition result: when the guard is true the Negated=false facts hold,
  /// when it is false the Negated=true facts hold. A guard built from p && q
  /// carries one fact per conjunct. recordPointerImplication also keys
  /// pointer variables here.
  using BoolGuardMap =
      llvm::DenseMap<PtrRef, llvm::SmallVector<ConditionResult, 2>>;
  BoolGuardMap BoolGuards;

  /// Pointer aliases: y = x records {y -> x}, meaning y holds the same value
  /// as x, so narrowing either one on a branch narrows the other. Chains are
  /// flattened: after z = y with {y -> x}, z maps directly to x.
  using AliasMap = llvm::DenseMap<const VarDecl *, const VarDecl *>;
  AliasMap Aliases;

  /// Local pointer copied from a member path: T *q = s->next stores
  /// {q -> s.next}. Narrowing either side narrows the other, so both
  /// if (s->next) { T *q = s->next; *q; } and T *q = s->next; if (q) *s->next
  /// are accepted. Dropped when q is reassigned or the path is invalidated.
  using MemberAliasMap = llvm::DenseMap<const VarDecl *, MemberAccessPath>;
  MemberAliasMap MemberAliases;

  /// Pointers to local pointers: T **pp = &p records {pp -> p}. A store
  /// through *pp can change p, so it drops p's narrowing. Dropped when pp is
  /// reassigned.
  using AddrOfTargetMap = llvm::DenseMap<const VarDecl *, const VarDecl *>;
  AddrOfTargetMap AddrOfTargets;

  void markNarrowed(const VarDecl *VD) {
    NarrowedVars.insert(VD);
    NullableVars.erase(VD);
  }
  void markNullable(const VarDecl *VD) {
    NarrowedVars.erase(VD);
    NullableVars.insert(VD);
    MustNullableVars.insert(VD);
  }
  void markNarrowed(const MemberAccessPath &P) {
    NarrowedMembers.insert(P);
    NullableMembers.erase(P);
  }
  void markNullable(const MemberAccessPath &P) {
    NarrowedMembers.erase(P);
    NullableMembers.insert(P);
    MustNullableMembers.insert(P);
  }

  bool isNarrowed(const PtrRef &R) const {
    return R.VD ? NarrowedVars.contains(R.VD)
                : NarrowedMembers.contains(*R.Path);
  }
  bool isNullable(const PtrRef &R) const {
    return R.VD ? NullableVars.contains(R.VD)
                : NullableMembers.contains(*R.Path);
  }
  bool isMustNullable(const PtrRef &R) const {
    return isNullable(R) && (R.VD ? MustNullableVars.contains(R.VD)
                                  : MustNullableMembers.contains(*R.Path));
  }
  void markNarrowed(const PtrRef &R) {
    if (R.VD)
      markNarrowed(R.VD);
    else
      markNarrowed(*R.Path);
  }
  void markNullable(const PtrRef &R) {
    if (R.VD)
      markNullable(R.VD);
    else
      markNullable(*R.Path);
  }
  /// Forget both the narrowed and the nullable fact about R.
  void clear(const PtrRef &R) {
    if (R.VD) {
      NarrowedVars.erase(R.VD);
      NullableVars.erase(R.VD);
    } else {
      NarrowedMembers.erase(*R.Path);
      NullableMembers.erase(*R.Path);
    }
  }

  bool operator==(const NullState &Other) const {
    return NarrowedVars == Other.NarrowedVars &&
           NarrowedMembers == Other.NarrowedMembers &&
           NullableVars == Other.NullableVars &&
           NullableMembers == Other.NullableMembers &&
           MustNullableVars == Other.MustNullableVars &&
           MustNullableMembers == Other.MustNullableMembers &&
           BoolGuards == Other.BoolGuards && Aliases == Other.Aliases &&
           MemberAliases == Other.MemberAliases &&
           AddrOfTargets == Other.AddrOfTargets;
  }
  bool operator!=(const NullState &Other) const { return !(*this == Other); }
};

} // end anonymous namespace

static NullState join(const NullState &A, const NullState &B) {
  NullState Result;
  // Narrowed = intersection: only narrowed if ALL paths agree.
  for (const auto *VD : A.NarrowedVars)
    if (B.NarrowedVars.contains(VD))
      Result.NarrowedVars.insert(VD);
  for (const auto &MK : A.NarrowedMembers)
    if (B.NarrowedMembers.contains(MK))
      Result.NarrowedMembers.insert(MK);
  // Nullable = union: if nullable on either path, it's nullable.
  for (const auto *VD : A.NullableVars)
    Result.NullableVars.insert(VD);
  for (const auto *VD : B.NullableVars)
    Result.NullableVars.insert(VD);
  for (const auto &Path : A.NullableMembers)
    Result.NullableMembers.insert(Path);
  for (const auto &Path : B.NullableMembers)
    Result.NullableMembers.insert(Path);
  for (const auto *VD : A.MustNullableVars)
    if (A.NullableVars.contains(VD) && B.NullableVars.contains(VD) &&
        B.MustNullableVars.contains(VD))
      Result.MustNullableVars.insert(VD);
  for (const auto &Path : A.MustNullableMembers)
    if (A.NullableMembers.contains(Path) && B.NullableMembers.contains(Path) &&
        B.MustNullableMembers.contains(Path))
      Result.MustNullableMembers.insert(Path);
  // Maps intersect with value equality: an entry survives only when both
  // sides map the key to the same fact.
  for (const auto &[Guard, GuardInfo] : A.BoolGuards) {
    auto It = B.BoolGuards.find(Guard);
    if (It != B.BoolGuards.end() && It->second == GuardInfo)
      Result.BoolGuards[Guard] = GuardInfo;
  }
  for (const auto &[AliasVD, TargetVD] : A.Aliases) {
    auto It = B.Aliases.find(AliasVD);
    if (It != B.Aliases.end() && It->second == TargetVD)
      Result.Aliases[AliasVD] = TargetVD;
  }
  for (const auto &[AliasVD, Path] : A.MemberAliases) {
    auto It = B.MemberAliases.find(AliasVD);
    if (It != B.MemberAliases.end() && It->second == Path)
      Result.MemberAliases[AliasVD] = Path;
  }
  for (const auto &[PtrPtrVD, TargetVD] : A.AddrOfTargets) {
    auto It = B.AddrOfTargets.find(PtrPtrVD);
    if (It != B.AddrOfTargets.end() && It->second == TargetVD)
      Result.AddrOfTargets[PtrPtrVD] = TargetVD;
  }
  // Narrowed wins over a stale nullable entry, so NullableVars does not
  // accumulate across fixpoint iterations.
  for (const auto *VD : Result.NarrowedVars)
    Result.NullableVars.erase(VD);
  for (const auto &Path : Result.NarrowedMembers)
    Result.NullableMembers.erase(Path);
  LLVM_DEBUG({
    llvm::dbgs() << "  join: narrowed=" << Result.NarrowedVars.size()
                 << " nullable=" << Result.NullableVars.size()
                 << " members=" << Result.NarrowedMembers.size()
                 << " aliases=" << Result.Aliases.size() << "\n";
  });
  return Result;
}

//===----------------------------------------------------------------------===//
// Expression unwrapping helpers
//===----------------------------------------------------------------------===//

/// Look through __builtin_expect(cond, ...) to cond.
static const Expr *unwrapBuiltinExpect(const Expr *E) {
  if (const auto *CE = dyn_cast<CallExpr>(E)) {
    if (const auto *Callee = CE->getDirectCallee()) {
      unsigned BuiltinID = Callee->getBuiltinID();
      if ((BuiltinID == Builtin::BI__builtin_expect ||
           BuiltinID == Builtin::BI__builtin_expect_with_probability) &&
          CE->getNumArgs() >= 1) {
        // IgnoreParenCasts, not ImpCasts: __builtin_expect((long)(p == 0), 0)
        // wraps the condition in an explicit integer cast.
        return CE->getArg(0)->IgnoreParenCasts();
      }
    }
  }
  return E;
}

/// Look through an OpaqueValueExpr (the shared operand of a GNU a ?: b)
/// to the expression it stands for.
static const Expr *stripOpaqueValue(const Expr *E) {
  if (const auto *OVE = dyn_cast<OpaqueValueExpr>(E))
    if (const Expr *Src = OVE->getSourceExpr())
      return Src->IgnoreParenImpCasts();
  return E;
}

/// Constant truth value of an integer/bool expression, if it has one.
static std::optional<bool> constantTruth(const Expr *E, ASTContext &Ctx) {
  E = E->IgnoreParenImpCasts();
  if (const auto *BL = dyn_cast<CXXBoolLiteralExpr>(E))
    return BL->getValue();
  if (E->getType()->isIntegerType())
    if (auto V = E->getIntegerConstantExpr(Ctx))
      return !V->isZero();
  return std::nullopt;
}

/// Strip explicit casts to bool so the underlying condition is visible.
/// glibc/libstdc++ define assert(expr) as
///   (static_cast<bool>(expr) ? void(0) : __assert_fail(...))
/// and IgnoreParenImpCasts() leaves that CXXStaticCastExpr in place, hiding
/// the tested pointer from narrowing.
static const Expr *ignoreExplicitBoolCast(const Expr *E) {
  while (const auto *CE = dyn_cast<ExplicitCastExpr>(E)) {
    if (!CE->getType()->isBooleanType())
      break;
    E = CE->getSubExpr()->IgnoreParenImpCasts();
  }
  return E;
}

/// Strip explicit casts whose source and result are both pointer types, so
/// *(int*)o still reaches the tracked VarDecl o (IgnoreParenImpCasts leaves
/// explicit casts in place). Integer->pointer and other casts produce a new
/// value with no tracked origin and are not stripped. Neither is dynamic_cast:
/// it returns null when the runtime type check fails, so a non-null source can
/// yield a null result and dynamic_cast<T*>(p)->f must still warn.
static const Expr *lookThroughPtrToPtrCasts(const Expr *E) {
  while (const auto *CE = dyn_cast<ExplicitCastExpr>(E)) {
    if (isa<CXXDynamicCastExpr>(CE))
      break;
    const Expr *Sub = CE->getSubExpr()->IgnoreParenImpCasts();
    if (!CE->getType()->isPointerType() || !Sub->getType()->isPointerType())
      break;
    E = Sub;
  }
  return E;
}

/// Strip explicit casts, stopping at dynamic_cast (which can yield null from
/// a non-null source, so provenance does not carry through it). Unlike
/// lookThroughPtrToPtrCasts this does not require pointer types on both
/// sides. *FoundCast is set when any explicit cast was seen.
static const Expr *stripNonDynamicCasts(const Expr *E,
                                        bool *FoundCast = nullptr) {
  while (const auto *CE = dyn_cast<ExplicitCastExpr>(E)) {
    if (FoundCast)
      *FoundCast = true;
    if (isa<CXXDynamicCastExpr>(CE))
      break;
    E = CE->getSubExpr()->IgnoreParenImpCasts();
  }
  return E;
}

/// Extract the rightmost leaf of a && / || chain. The CFG gives each operand
/// its own block but every such block carries the full a && b && c as its
/// terminator; the leaf actually evaluated in the last block is the RHS.
static const Expr *getTerminalCondition(const Expr *E) {
  E = E->IgnoreParenImpCasts();
  if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
    if (BO->getOpcode() == BO_LAnd || BO->getOpcode() == BO_LOr)
      return getTerminalCondition(BO->getRHS());
  }
  return E;
}

//===----------------------------------------------------------------------===//
// Type and stdlib helpers
//===----------------------------------------------------------------------===//

/// Whether a declared type may be null: an explicit _Nullable always is, and
/// _Null_unspecified is only under -fnullability-default=nullable (the
/// nonnull default treats it as non-null).
static bool isNullableType(QualType Ty, NullabilityKind Default) {
  NullabilityKindOrNone Nullability = Ty->getNullability();
  if (!Nullability)
    return false;
  if (*Nullability == NullabilityKind::Nullable)
    return true;
  if (*Nullability == NullabilityKind::Unspecified &&
      Default == NullabilityKind::Nullable)
    return true;
  return false;
}

/// Template argument \p Idx of the sugared specialization type underlying
/// \p Ty (references and one level of pointer stripped), or a null QualType.
/// Reads the sugar rather than the ClassTemplateSpecializationDecl because
/// the latter strips nullability from its TemplateArguments.
static QualType getSugaredTemplateArgType(QualType Ty, unsigned Idx) {
  Ty = Ty.getNonReferenceType();
  if (Ty->isPointerType())
    Ty = Ty->getPointeeType();
  const auto *TST = Ty->getAs<TemplateSpecializationType>();
  if (!TST)
    return QualType();
  auto Args = TST->template_arguments();
  if (Idx >= Args.size() || Args[Idx].getKind() != TemplateArgument::Type)
    return QualType();
  return Args[Idx].getAsType();
}

/// Recover nullability for a field whose type came from a template argument:
/// Box<int*_Nullable>::val has type int* in the AST. The sugar survives only
/// on the base expression's type, since ClassTemplateSpecializationDecl strips
/// it from its TemplateArguments.
static QualType getTemplateArgTypeForField(const MemberExpr *ME) {
  const auto *FD = dyn_cast<FieldDecl>(ME->getMemberDecl());
  if (!FD)
    return QualType();
  const auto *RD = dyn_cast<CXXRecordDecl>(FD->getParent());
  if (!RD)
    return QualType();
  const auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RD);
  if (!CTSD)
    return QualType();
  CXXRecordDecl *Pattern = CTSD->getSpecializedTemplate()->getTemplatedDecl();
  for (const auto *PatternField : Pattern->fields()) {
    if (PatternField->getDeclName() != FD->getDeclName())
      continue;
    const auto *TTPT = PatternField->getType()->getAs<TemplateTypeParmType>();
    if (!TTPT || TTPT->getDepth() != 0)
      return QualType();
    return getSugaredTemplateArgType(
        ME->getBase()->IgnoreParenImpCasts()->getType(), TTPT->getIndex());
  }
  return QualType();
}

/// Like getTemplateArgTypeForField, for method return types: when
/// Container<int*_Nullable> has method T get(), the instantiated return type
/// is int* but the template argument carries _Nullable.
static QualType
getTemplateArgTypeForMethodReturn(const CXXMemberCallExpr *MCE) {
  const auto *MD = MCE->getMethodDecl();
  if (!MD)
    return QualType();
  // Case 1: member function template (e.g. S::get<int*_Nullable>()).
  // The template args are on the function itself. Get the sugared args
  // from the MemberExpr in the call, not from the FunctionDecl (which
  // strips sugar on implicit instantiations).
  if (const auto *FT = MD->getPrimaryTemplate()) {
    const FunctionDecl *Pattern = FT->getTemplatedDecl();
    const auto *TTPT = Pattern->getReturnType()->getAs<TemplateTypeParmType>();
    if (TTPT && TTPT->getDepth() == 0) {
      unsigned ArgIdx = TTPT->getIndex();
      const auto *ME = dyn_cast<MemberExpr>(MCE->getCallee());
      if (ME && ME->hasExplicitTemplateArgs()) {
        auto Args = ME->template_arguments();
        if (ArgIdx < Args.size() &&
            Args[ArgIdx].getArgument().getKind() == TemplateArgument::Type)
          return Args[ArgIdx].getArgument().getAsType();
      }
    }
  }

  // Case 2: class template method (e.g. Container<int*_Nullable>::get()).
  const auto *RD = dyn_cast<CXXRecordDecl>(MD->getParent());
  if (!RD)
    return QualType();
  const auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RD);
  if (!CTSD)
    return QualType();
  const FunctionDecl *PatternMethod = MD->getInstantiatedFromMemberFunction();
  if (!PatternMethod) {
    // Fallback for methods not directly instantiated (e.g. inherited).
    CXXRecordDecl *Pattern = CTSD->getSpecializedTemplate()->getTemplatedDecl();
    for (const auto *PM : Pattern->methods()) {
      if (PM->getDeclName() == MD->getDeclName() &&
          PM->getNumParams() == MD->getNumParams()) {
        PatternMethod = PM;
        break;
      }
    }
  }
  if (PatternMethod) {
    const auto *TTPT =
        PatternMethod->getReturnType()->getAs<TemplateTypeParmType>();
    if (!TTPT || TTPT->getDepth() != 0)
      return QualType();
    // IgnoreParenImpCasts to see through the const-qualification cast that
    // strips template argument sugar.
    const Expr *Obj = MCE->getImplicitObjectArgument();
    if (!Obj)
      return QualType();
    return getSugaredTemplateArgType(Obj->IgnoreParenImpCasts()->getType(),
                                     TTPT->getIndex());
  }
  return QualType();
}

/// Whether a type is explicitly _Nullable. Unannotated types that are merely
/// defaulted to nullable do not count, so evidence never infers _Nullable
/// from an unannotated source.
static bool isExplicitlyNullableType(QualType Ty) {
  NullabilityKindOrNone Nullability = Ty->getNullability();
  return Nullability && *Nullability == NullabilityKind::Nullable;
}

/// Whether a declared type is explicitly _Nonnull.
static bool isNonnullType(QualType Ty) {
  NullabilityKindOrNone Nullability = Ty->getNullability();
  return Nullability && *Nullability == NullabilityKind::NonNull;
}

struct ExplicitCastInfo {
  const ExplicitCastExpr *DefinitiveCast;
  const Expr *Operand;

  bool isNonnull() const {
    return DefinitiveCast && isNonnullType(DefinitiveCast->getType());
  }
};

/// Walk a chain of explicit casts, returning the outermost cast with a
/// definite (not unspecified) nullability and the innermost operand.
static ExplicitCastInfo getExplicitCastInfo(const Expr *E) {
  const ExplicitCastExpr *DefinitiveCast = nullptr;
  while (const auto *CE = dyn_cast<ExplicitCastExpr>(E)) {
    if (!DefinitiveCast)
      if (auto Current = CE->getType()->getNullability())
        if (*Current != NullabilityKind::Unspecified)
          DefinitiveCast = CE;
    E = CE->getSubExpr()->IgnoreParenImpCasts();
  }
  return {DefinitiveCast, E};
}

/// Declared type of the variable or field a smart pointer expression names,
/// with any reference stripped, or a null QualType. Overload resolution on
/// operator->/operator* strips nullability from the expression's own type, so
/// only the declaration still carries it. For a reference the annotation sits
/// on the referenced type (const std::shared_ptr<T> _Nonnull &), and a local
/// reference that carries none takes its referent's, as its flow facts do
/// (see resolveSmartPtrReference).
static QualType getSmartPointerDeclaredType(const Expr *E) {
  E = E->IgnoreParenImpCasts();
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      QualType Ty = VD->getType().getNonReferenceType();
      if (!Ty->getNullability())
        Ty = resolveSmartPtrReference(VD)->getType().getNonReferenceType();
      return Ty;
    }
  } else if (const auto *ME = dyn_cast<MemberExpr>(E)) {
    if (const auto *FD = dyn_cast<FieldDecl>(ME->getMemberDecl()))
      return FD->getType().getNonReferenceType();
  }
  return QualType();
}

static bool isSmartPointerDeclaredNonnull(const Expr *E) {
  QualType Ty = getSmartPointerDeclaredType(E);
  return !Ty.isNull() && isNonnullType(Ty);
}

static bool isSmartPointerDeclaredNullable(const Expr *E) {
  QualType Ty = getSmartPointerDeclaredType(E);
  return !Ty.isNull() && isExplicitlyNullableType(Ty);
}

/// Check if a type is std::unique_ptr, std::shared_ptr, or std::weak_ptr.
/// Uses getAsCXXRecordDecl() which operates on the canonical type, so
/// type aliases (using/typedef) are handled. Does not match non-std
/// smart pointers (e.g. boost::shared_ptr).
static bool isSmartPointerType(QualType Ty) {
  const auto *RD = Ty->getAsCXXRecordDecl();
  if (!RD)
    return false;
  const auto *DC = RD->getDeclContext();
  if (!DC || !DC->isStdNamespace())
    return false;
  StringRef Name = RD->getName();
  return Name == "unique_ptr" || Name == "shared_ptr" || Name == "weak_ptr";
}

/// Whether Obj, the object a member call or operator acts on, is a std smart
/// pointer: by its own type, or by a class on the derived-to-base path the
/// call took to an inherited member. libstdc++ declares shared_ptr's
/// operator->, operator*, get(), operator bool and reset() on base classes
/// (__shared_ptr_access, __shared_ptr), and a user class deriving from a std
/// smart pointer reaches that pointer's members the same way. Neither the
/// type at the call nor the type under the casts covers both: for a class
/// deriving from libstdc++'s shared_ptr, one is __shared_ptr_access and the
/// other is the user class.
static bool isSmartPointerObject(const Expr *Obj) {
  for (const Expr *E = Obj; E;) {
    if (isSmartPointerType(E->getType()))
      return true;
    const auto *ICE = dyn_cast<ImplicitCastExpr>(E->IgnoreParens());
    if (!ICE)
      return false;
    if (ICE->getCastKind() == CK_DerivedToBase ||
        ICE->getCastKind() == CK_UncheckedDerivedToBase)
      for (const CXXBaseSpecifier *Base : ICE->path())
        if (isSmartPointerType(Base->getType()))
          return true;
    E = ICE->getSubExpr();
  }
  return false;
}

/// Strip implicit wrappers that real standard library headers introduce
/// around expressions (ExprWithCleanups, CXXBindTemporaryExpr,
/// MaterializeTemporaryExpr) plus the usual parens and implicit casts.
static const Expr *unwrapImplicitWrappers(const Expr *E) {
  while (true) {
    E = E->IgnoreParenImpCasts();
    if (const auto *EWC = dyn_cast<ExprWithCleanups>(E))
      E = EWC->getSubExpr();
    else if (const auto *BTE = dyn_cast<CXXBindTemporaryExpr>(E))
      E = BTE->getSubExpr();
    else if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
      E = MTE->getSubExpr();
    else
      break;
  }
  return E;
}

/// The std function CE calls directly, when it is one named Name.
static bool isStdCallTo(const CallExpr *CE, StringRef Name) {
  const FunctionDecl *Callee = CE ? CE->getDirectCallee() : nullptr;
  return Callee && Callee->isInStdNamespace() &&
         Callee->getDeclName().isIdentifier() && Callee->getName() == Name;
}

/// The argument of std::static_pointer_cast, const_pointer_cast or
/// reinterpret_pointer_cast (whose result is null exactly when the argument
/// is), or nullptr when E is not such a call. dynamic_pointer_cast is not
/// one: it yields null when the runtime check fails.
static const Expr *getNullPreservingPointerCastArg(const Expr *E) {
  const auto *CE = dyn_cast<CallExpr>(E);
  if (!CE || CE->getNumArgs() != 1)
    return nullptr;
  if (isStdCallTo(CE, "static_pointer_cast") ||
      isStdCallTo(CE, "const_pointer_cast") ||
      isStdCallTo(CE, "reinterpret_pointer_cast"))
    return CE->getArg(0);
  return nullptr;
}

/// Check if a smart pointer is constructed from a provably non-null source:
/// make_unique/make_shared, a constructor taking a new-expression, or a call
/// whose declared return type is _Nonnull, possibly through a null-preserving
/// pointer cast. The callee's declaration is read because the call
/// expression's type can lose the annotation.
static bool isNonnullSmartPtrInit(const Expr *E) {
  E = unwrapImplicitWrappers(E);
  if (const Expr *Arg = getNullPreservingPointerCastArg(E))
    return isNonnullSmartPtrInit(Arg);
  if (const auto *CE = dyn_cast<CallExpr>(E))
    if (const FunctionDecl *Callee = CE->getDirectCallee())
      if (isNonnullType(Callee->getReturnType().getNonReferenceType()))
        return true;
  if (const auto *CE = dyn_cast<CXXConstructExpr>(E)) {
    if (CE->getNumArgs() == 1)
      return isNonnullSmartPtrInit(CE->getArg(0));
  }
  // unique_ptr<T>(new T()) wraps the constructor in a functional cast node.
  if (const auto *FCE = dyn_cast<CXXFunctionalCastExpr>(E))
    return isNonnullSmartPtrInit(FCE->getSubExpr());
  // new T(): throwing operator new never returns null.
  if (const auto *NE = dyn_cast<CXXNewExpr>(E))
    return !NE->shouldNullCheckAllocation();
  if (const auto *CE = dyn_cast<CallExpr>(E)) {
    if (const auto *Callee = CE->getDirectCallee()) {
      const auto *DC = Callee->getDeclContext();
      if (DC && DC->isStdNamespace() && Callee->getDeclName().isIdentifier()) {
        StringRef Name = Callee->getName();
        return Name == "make_unique" || Name == "make_shared";
      }
    }
  }
  return false;
}

/// Whether a smart pointer VarDecl is initialized as *it with it =
/// container.begin() (range-for desugaring or a manual loop) on a container
/// whose element type carries _Nonnull. Instantiation strips _Nonnull from
/// non-pointer types like unique_ptr, so the container's sugar type is the
/// only place the annotation survives.
static bool isInitFromNonnullContainerElement(const VarDecl *VD) {
  if (!VD->hasInit())
    return false;
  const Expr *Init = unwrapImplicitWrappers(VD->getInit());

  const auto *OpCall = dyn_cast<CXXOperatorCallExpr>(Init);
  if (!OpCall || OpCall->getOperator() != OO_Star || OpCall->getNumArgs() < 1)
    return false;

  const Expr *IterExpr = OpCall->getArg(0)->IgnoreParenImpCasts();
  const auto *IterDRE = dyn_cast<DeclRefExpr>(IterExpr);
  if (!IterDRE)
    return false;
  const auto *IterVD = dyn_cast<VarDecl>(IterDRE->getDecl());
  if (!IterVD || !IterVD->hasInit())
    return false;

  const Expr *IterInit = unwrapImplicitWrappers(IterVD->getInit());
  const auto *BeginCall = dyn_cast<CXXMemberCallExpr>(IterInit);
  if (!BeginCall)
    return false;

  // Use the declared type: the implicit const cast on .begin()'s object arg
  // strips the sugar.
  const Expr *ObjArg = BeginCall->getImplicitObjectArgument();
  if (!ObjArg)
    return false;
  const Expr *ContainerExpr = ObjArg->IgnoreParenImpCasts();
  QualType ContainerType;
  if (const auto *DRE = dyn_cast<DeclRefExpr>(ContainerExpr)) {
    if (const auto *CVD = dyn_cast<VarDecl>(DRE->getDecl()))
      ContainerType = CVD->getType().getNonReferenceType();
  } else if (const auto *ME = dyn_cast<MemberExpr>(ContainerExpr)) {
    if (const auto *FD = dyn_cast<FieldDecl>(ME->getMemberDecl()))
      ContainerType = FD->getType().getNonReferenceType();
  }
  // A pointer to a container (auto it = v->begin()) is rejected:
  // getSugaredTemplateArgType would look through the pointer.
  if (ContainerType.isNull() || ContainerType->isPointerType())
    return false;

  QualType ElemType = getSugaredTemplateArgType(ContainerType, 0);
  if (ElemType.isNull())
    return false;
  return isSmartPointerType(ElemType) && isNonnullType(ElemType);
}

/// Allowlist of STL methods whose unannotated pointer returns are
/// contractually non-null (vector/string/array/span data() and iterators,
/// optional::operator->). Overlay headers cannot redeclare these, so they
/// are matched by name here.
static bool isStlNonnullReturnCall(const CallExpr *CE) {
  const auto *MCE = dyn_cast<CXXMemberCallExpr>(CE);
  if (!MCE)
    return false;
  const auto *MD = MCE->getMethodDecl();
  if (!MD)
    return false;
  if (!MD->getReturnType()->isPointerType())
    return false;
  const auto *RD = MD->getParent();
  if (!RD)
    return false;
  const auto *DC = RD->getDeclContext();
  if (!DC || !DC->isStdNamespace())
    return false;
  StringRef ClassName = RD->getName();

  const auto &DeclName = MD->getDeclName();
  StringRef MethodName;
  bool IsArrowOp = false;
  if (DeclName.isIdentifier()) {
    MethodName = MD->getName();
  } else if (DeclName.getNameKind() == DeclarationName::CXXOperatorName) {
    IsArrowOp = DeclName.getCXXOverloadedOperator() == OO_Arrow;
  }

  // std::vector<T>::data(), begin(), end().
  if (ClassName == "vector")
    return MethodName == "data" || MethodName == "begin" || MethodName == "end";

  // std::basic_string<T> (std::string, std::wstring, ...): c_str(), data(),
  // begin(), end().
  if (ClassName == "basic_string")
    return MethodName == "c_str" || MethodName == "data" ||
           MethodName == "begin" || MethodName == "end";

  // std::basic_string_view<T>::begin(), end(). data() is deliberately absent:
  // a string_view can hold nullptr.
  if (ClassName == "basic_string_view")
    return MethodName == "begin" || MethodName == "end";

  // std::optional<T>::operator->(): UB when empty, so callers assert a value.
  if (ClassName == "optional")
    return IsArrowOp;

  // A non-empty std::array owns inline storage, so these pointers are non-null.
  // std::array<T, 0> is excluded because its data()/iterators may be null.
  if (ClassName == "array" &&
      (MethodName == "data" || MethodName == "begin" || MethodName == "end")) {
    const auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(RD);
    if (!Spec || Spec->getTemplateArgs().size() < 2)
      return false;
    const TemplateArgument &Extent = Spec->getTemplateArgs()[1];
    return Extent.getKind() == TemplateArgument::Integral &&
           !Extent.getAsIntegral().isZero();
  }

  // An empty std::span may return a null pointer, but spans are almost always
  // non-empty, so these accessors are treated as non-null rather than warning
  // on every use.
  if (ClassName == "span")
    return MethodName == "data" || MethodName == "begin" || MethodName == "end";

  return false;
}

/// C library free functions that return null on failure or not-found
/// (malloc, fopen, getenv, strchr, ...). Their results are provably _Nullable
/// regardless of annotations, so unchecked dereferences always warn. Only
/// free functions at global or std scope match.
static bool isLibcNullableReturnCall(const CallExpr *CE) {
  if (isa<CXXMemberCallExpr>(CE))
    return false;
  const FunctionDecl *FD = CE->getDirectCallee();
  if (!FD)
    return false;
  if (!FD->getReturnType()->isPointerType())
    return false;
  if (isa<CXXMethodDecl>(FD))
    return false;
  const auto &DeclName = FD->getDeclName();
  if (!DeclName.isIdentifier())
    return false;
  // Only the real C library functions match: those at global scope or in std
  // (std::malloc from <cstdlib>). A user function that happens to share the
  // name (namespace my { int *malloc(); }) is left alone, so a name collision
  // cannot cause false positives.
  const DeclContext *DC = FD->getDeclContext()->getRedeclContext();
  if (!DC->isTranslationUnit() && !DC->isStdNamespace())
    return false;
  StringRef Name = FD->getName();
  // Grouped by header (stdlib.h, stdio.h, string.h, ...).
  return llvm::StringSwitch<bool>(Name)
      .Cases({"malloc", "calloc", "realloc", "aligned_alloc"}, true)
      .Cases({"fopen", "freopen", "tmpfile"}, true)
      .Cases({"getenv", "strtok"}, true)
      .Cases({"strstr", "strchr", "strrchr", "strpbrk"}, true)
      .Cases({"memchr", "bsearch"}, true)
      .Cases({"tmpnam", "setlocale"}, true)
      .Default(false);
}

/// The tracked pointer E names, if E is a std smart pointer object (see
/// isSmartPointerObject). The expression's type is tested rather than the
/// declaration's: a reference (const std::shared_ptr<T> &p, a reference
/// member) declares a reference type but names the smart pointer itself, and
/// a class deriving from a std smart pointer shows up through the
/// derived-to-base cast of an inherited member's object argument.
static std::optional<PtrRef> smartPtrRef(const Expr *E) {
  if (!isSmartPointerObject(E))
    return std::nullopt;
  return PtrRef::fromExpr(E);
}

/// E with implicit wrappers, single-argument (converting) constructors and
/// null-preserving pointer casts looked through, down to the value whose
/// nullness E has.
static const Expr *lookThroughNullPreservingConversions(const Expr *E) {
  E = unwrapImplicitWrappers(E);
  while (true) {
    if (const auto *CCE = dyn_cast<CXXConstructExpr>(E)) {
      if (CCE->getNumArgs() != 1)
        return E;
      E = unwrapImplicitWrappers(CCE->getArg(0));
    } else if (const Expr *Arg = getNullPreservingPointerCastArg(E)) {
      E = unwrapImplicitWrappers(Arg);
    } else {
      return E;
    }
  }
}

/// The smart pointer a copy (T t = s, T t(s), t = s) reads from, seen through
/// converting constructors (shared_ptr<Base> b = derived) and null-preserving
/// casts (std::static_pointer_cast<D>(s)). Null when E is not a plain read of
/// a smart pointer variable or member.
static const Expr *smartPtrCopySource(const Expr *E) {
  E = lookThroughNullPreservingConversions(E);
  return smartPtrRef(E) ? E : nullptr;
}

/// The call (or operator[]) whose result the smart pointer object Obj is, if
/// that result may be null by contract: the callee's declared return type is
/// _Nullable (read from the declaration, since overload resolution strips it
/// from the object expression), or the callee is std::dynamic_pointer_cast.
/// Such a temporary has no identity to narrow, so an unannotated return is
/// not judged by the mode default.
static const CallExpr *getNullableSmartPtrCallResult(const Expr *Obj) {
  const auto *CE = dyn_cast<CallExpr>(unwrapImplicitWrappers(Obj));
  if (!CE)
    return nullptr;
  if (isStdCallTo(CE, "dynamic_pointer_cast"))
    return CE;
  const FunctionDecl *Callee = CE->getDirectCallee();
  if (Callee &&
      isExplicitlyNullableType(Callee->getReturnType().getNonReferenceType()))
    return CE;
  return nullptr;
}

/// Whether this std::move(sp) initializes or is assigned to a smart pointer
/// (auto x = std::move(y); or x = std::move(y)), or binds a reference to one
/// (auto &&r = std::move(y), which moves nothing). The transfer handler needs
/// the source's pre-move state, but the CFG visits the inner call before the
/// enclosing DeclStmt or operator=, so the context is recovered from the
/// function-scoped ParentMap rather than passed down.
static bool isStdMoveInsideSmartPtrTransferCtx(const CallExpr *CE,
                                               const ParentMap &PM) {
  // A VarDecl initializer's ParentMap parent is the DeclStmt itself. Match the
  // declarator whose init we walked up from; wrapper nodes can make that
  // pointer differ from VD->getInit(), so fall back to the sole declarator.
  auto isSmartPtrInitDecl = [](const DeclStmt *DS, const Stmt *Init) -> bool {
    const VarDecl *Sole = nullptr;
    unsigned NumInited = 0;
    for (const auto *D : DS->decls()) {
      if (const auto *VD = dyn_cast<VarDecl>(D)) {
        if (VD->getInit() == Init)
          return isSmartPointerType(VD->getType().getNonReferenceType());
        if (VD->hasInit()) {
          ++NumInited;
          Sole = VD;
        }
      }
    }
    if (NumInited == 1)
      return isSmartPointerType(Sole->getType().getNonReferenceType());
    return false;
  };

  const Stmt *Child = CE;
  for (const Stmt *S = PM.getParent(CE); S; S = PM.getParent(S)) {
    if (const auto *DS = dyn_cast<DeclStmt>(S))
      return isSmartPtrInitDecl(DS, Child);
    if (const auto *OCE = dyn_cast<CXXOperatorCallExpr>(S)) {
      if (OCE->getOperator() == OO_Equal && OCE->getNumArgs() >= 2) {
        if (smartPtrRef(OCE->getArg(0)))
          return true;
      }
      return false;
    }
    // std::static_pointer_cast<D>(std::move(sp)) moves sp into the result,
    // so the transfer reaches through the cast.
    const auto *E = dyn_cast<Expr>(S);
    if (isa<ExprWithCleanups>(S) || isa<CXXBindTemporaryExpr>(S) ||
        isa<MaterializeTemporaryExpr>(S) || isa<ImplicitCastExpr>(S) ||
        isa<ParenExpr>(S) || isa<CXXConstructExpr>(S) ||
        isa<CXXFunctionalCastExpr>(S) ||
        (E && getNullPreservingPointerCastArg(E))) {
      Child = S;
      continue;
    }
    return false;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Condition analysis
//===----------------------------------------------------------------------===//

/// The smart pointer receiver of an sp.get() call, or nullptr when CE is
/// not such a call.
static const Expr *smartPtrGetReceiver(const CallExpr *CE) {
  const auto *MCE = dyn_cast<CXXMemberCallExpr>(CE);
  if (!MCE)
    return nullptr;
  const auto *MD = MCE->getMethodDecl();
  if (!MD || !MD->getDeclName().isIdentifier() || MD->getName() != "get")
    return nullptr;
  const Expr *Obj = MCE->getImplicitObjectArgument();
  if (!isSmartPointerObject(Obj))
    return nullptr;
  return Obj;
}

static std::optional<PtrRef> smartPtrGetRef(const Expr *E) {
  const auto *CE = dyn_cast<CallExpr>(E->IgnoreParenImpCasts());
  const Expr *Obj = CE ? smartPtrGetReceiver(CE) : nullptr;
  if (!Obj)
    return std::nullopt;
  return smartPtrRef(Obj);
}

// Forward declaration: decomposeChain calls analyzeCondition on leaves.
static void
analyzeCondition(const Expr *Cond, ASTContext &Ctx,
                 SmallVectorImpl<ConditionResult> &Results,
                 const NullState::BoolGuardMap *BoolGuards = nullptr);

/// Flatten a chain of Op (BO_LAnd or BO_LOr) and analyze each leaf. The &&
/// form handles !(A && B && C) and guard initializers like ok = a && b; the
/// || form narrows every operand on the false edge of if (A || B).
static void decomposeChain(const Expr *E, BinaryOperatorKind Op,
                           ASTContext &Ctx,
                           SmallVectorImpl<ConditionResult> &Results,
                           const NullState::BoolGuardMap *BoolGuards) {
  E = E->IgnoreParenImpCasts();
  if (const auto *EWC = dyn_cast<ExprWithCleanups>(E))
    E = EWC->getSubExpr()->IgnoreParenImpCasts();
  if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
    if (BO->getOpcode() == Op) {
      decomposeChain(BO->getLHS(), Op, Ctx, Results, BoolGuards);
      decomposeChain(BO->getRHS(), Op, Ctx, Results, BoolGuards);
      return;
    }
  }
  analyzeCondition(E, Ctx, Results, BoolGuards);
}

/// Handle !(A && B). The CFG merges the && operand blocks before the if,
/// losing per-operand narrowing, so every operand is narrowed on the edge
/// where !(A && B) is false, since all of them held there. Returns true when
/// E was handled.
static bool analyzeNegatedAnd(const Expr *E, bool Negated, ASTContext &Ctx,
                              SmallVectorImpl<ConditionResult> &Results,
                              const NullState::BoolGuardMap *BoolGuards) {
  if (Negated) {
    if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
      if (BO->getOpcode() == BO_LAnd) {
        // Collect into a local vector: Results may already hold leaves from
        // an outer decomposeChain (if (x && !(a && b))), and the filtering
        // below must not touch them.
        SmallVector<ConditionResult, 4> AndResults;
        decomposeChain(BO, BO_LAnd, Ctx, AndResults, BoolGuards);
        // Keep only sub-conditions where the pointer is non-null when the
        // sub-condition is true (Negated=false). Flip to Negated=true so
        // narrowing lands on the false edge of the outer !.
        llvm::erase_if(AndResults,
                       [](const ConditionResult &CR) { return CR.Negated; });
        for (auto &CR : AndResults) {
          CR.Negated = true;
          Results.push_back(std::move(CR));
        }
        return true;
      }
    }
  }
  return false;
}

/// p != nullptr / p == nullptr on a raw pointer, and flag == true / flag != 0
/// on a guard variable. Returns true when BO was such a comparison.
static bool analyzeNullCompare(const BinaryOperator *BO, bool Negated,
                               ASTContext &Ctx,
                               SmallVectorImpl<ConditionResult> &Results,
                               const NullState::BoolGuardMap *BoolGuards) {
  if (BO->getOpcode() == BO_NE || BO->getOpcode() == BO_EQ) {
    const Expr *LHS = BO->getLHS()->IgnoreParenImpCasts();
    const Expr *RHS = BO->getRHS()->IgnoreParenImpCasts();
    bool EqNegated = Negated;
    if (BO->getOpcode() == BO_EQ)
      EqNegated = !EqNegated;

    // A guard compared against a constant (flag == true, flag != 0,
    // flag == false). The stored facts flip when the comparison holding
    // means the guard is false.
    if (BoolGuards) {
      for (auto [GuardSide, ConstSide] :
           {std::pair{LHS, RHS}, std::pair{RHS, LHS}}) {
        auto Guard = PtrRef::fromExpr(GuardSide);
        if (!Guard || !Guard->getType()->isIntegerType())
          continue;
        auto It = BoolGuards->find(*Guard);
        if (It == BoolGuards->end())
          continue;
        std::optional<bool> CV = constantTruth(ConstSide, Ctx);
        if (!CV)
          continue;
        // EqNegated means the comparison is effectively == (a bare ==,
        // or != under an odd number of !), so it holds when the guard
        // equals CV; otherwise it holds when the guard equals !CV.
        bool GuardTruthWhenTrue = EqNegated ? *CV : !*CV;
        bool Flip = !GuardTruthWhenTrue;
        for (ConditionResult CR : It->second) {
          CR.Negated = CR.Negated != Flip;
          Results.push_back(std::move(CR));
        }
        return true;
      }
    }

    // IgnoreParenCasts, not ImpCasts: C spells null constants as
    // (T *)0 and that explicit cast is not a null pointer constant in
    // Clang's C mode.
    bool LHSIsNull = LHS->IgnoreParenCasts()->isNullPointerConstant(
        Ctx, Expr::NPC_ValueDependentIsNotNull);
    bool RHSIsNull = RHS->IgnoreParenCasts()->isNullPointerConstant(
        Ctx, Expr::NPC_ValueDependentIsNotNull);

    if (LHSIsNull || RHSIsNull) {
      const Expr *PtrExpr = LHSIsNull ? RHS : LHS;

      // An assignment in the condition, (p = f()) != nullptr, narrows p.
      if (const auto *AssignBO = dyn_cast<BinaryOperator>(PtrExpr)) {
        if (AssignBO->getOpcode() == BO_Assign)
          PtrExpr = AssignBO->getLHS()->IgnoreParenImpCasts();
      }

      // As on the dereference side, (T*)p != nullptr narrows p.
      PtrExpr = lookThroughPtrToPtrCasts(PtrExpr);

      if (auto R = smartPtrGetRef(PtrExpr)) {
        Results.push_back({std::move(*R), EqNegated});
        return true;
      }

      if (auto R = PtrRef::fromExpr(PtrExpr))
        if (R->getType()->isPointerType())
          Results.push_back({std::move(*R), EqNegated});
    }
    return true;
  }
  return false;
}

/// sp != nullptr / sp == nullptr through an overloaded operator on a smart
/// pointer. Returns true when OCE was such a comparison.
static bool
analyzeSmartPtrNullCompare(const CXXOperatorCallExpr *OCE, bool Negated,
                           ASTContext &Ctx,
                           SmallVectorImpl<ConditionResult> &Results) {
  auto OpKind = OCE->getOperator();
  if ((OpKind == OO_ExclaimEqual || OpKind == OO_EqualEqual) &&
      OCE->getNumArgs() == 2) {
    const Expr *LHS = OCE->getArg(0)->IgnoreParenImpCasts();
    const Expr *RHS = OCE->getArg(1)->IgnoreParenImpCasts();

    bool LHSIsNull =
        LHS->isNullPointerConstant(Ctx, Expr::NPC_ValueDependentIsNotNull);
    bool RHSIsNull =
        RHS->isNullPointerConstant(Ctx, Expr::NPC_ValueDependentIsNotNull);

    if (LHSIsNull || RHSIsNull) {
      // The operand as passed, so its type and any derived-to-base cast are
      // still visible to smartPtrRef.
      const Expr *PtrExpr = OCE->getArg(LHSIsNull ? 1 : 0);
      bool EqNegated = Negated;
      if (OpKind == OO_EqualEqual)
        EqNegated = !EqNegated;

      if (auto R = smartPtrRef(PtrExpr))
        Results.push_back({std::move(*R), EqNegated});
    }
    return true;
  }
  return false;
}

/// if (sp) on a smart pointer: an implicit operator bool() call.
static void
analyzeSmartPtrBoolConversion(const CXXMemberCallExpr *MCE, bool Negated,
                              SmallVectorImpl<ConditionResult> &Results) {
  if (const auto *CD =
          dyn_cast_or_null<CXXConversionDecl>(MCE->getMethodDecl())) {
    if (CD->getConversionType()->isBooleanType()) {
      if (auto R = smartPtrRef(MCE->getImplicitObjectArgument())) {
        Results.push_back({std::move(*R), Negated});
        return;
      }
    }
  }
}

/// Turn a branch condition into ConditionResult facts (see the file
/// overview). The CFG gives each && and || operand its own block, but a ||
/// operand that creates a temporary with a destructor (f() == nullptr on a
/// unique_ptr) adds cleanup blocks that merge the paths before the IfStmt, so
/// decomposeChain re-narrows every operand on the IfStmt's false edge.
static void analyzeCondition(const Expr *Cond, ASTContext &Ctx,
                             SmallVectorImpl<ConditionResult> &Results,
                             const NullState::BoolGuardMap *BoolGuards) {
  if (!Cond)
    return;

  const Expr *E = stripOpaqueValue(Cond->IgnoreParenImpCasts());

  // Peel !, __builtin_expect, explicit casts to bool and C++20 rewritten
  // comparisons, in any order. With only operator==(const shared_ptr &,
  // nullptr_t) declared (libc++ in C++20), sp != nullptr is a
  // CXXRewrittenBinaryOperator whose semantic form is !(sp == nullptr), and
  // assertion macros wrap it further: !(sp != nullptr),
  // __builtin_expect(!!(!(sp != nullptr)), 0).
  bool Negated = false;
  while (true) {
    const Expr *Next;
    if (const auto *RBO = dyn_cast<CXXRewrittenBinaryOperator>(E)) {
      Next = RBO->getSemanticForm();
    } else if (const auto *UO = dyn_cast<UnaryOperator>(E);
               UO && UO->getOpcode() == UO_LNot) {
      Negated = !Negated;
      Next = UO->getSubExpr();
    } else {
      Next = unwrapBuiltinExpect(E);
    }
    Next = ignoreExplicitBoolCast(Next->IgnoreParenImpCasts());
    if (Next == E)
      break;
    E = Next;
  }

  // See through pointer-to-pointer casts, as the dereference side does, so
  // if ((T*)p) { *p; } narrows p.
  E = lookThroughPtrToPtrCasts(E);

  if (analyzeNegatedAnd(E, Negated, Ctx, Results, BoolGuards))
    return;

  if (const auto *BO = dyn_cast<BinaryOperator>(E))
    if (analyzeNullCompare(BO, Negated, Ctx, Results, BoolGuards))
      return;

  // On a smart pointer, sp != nullptr is a CXXOperatorCallExpr.
  if (const auto *OCE = dyn_cast<CXXOperatorCallExpr>(E))
    if (analyzeSmartPtrNullCompare(OCE, Negated, Ctx, Results))
      return;

  if (const auto *UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() == UO_Deref) {
      const Expr *SubExpr = UO->getSubExpr()->IgnoreParenImpCasts();
      if (const auto *DRE = dyn_cast<DeclRefExpr>(SubExpr)) {
        if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
          if (VD->getType()->isPointerType()) {
            Results.push_back({PtrRef{VD, std::nullopt}, Negated});
            return;
          }
        }
      }
    }
  }

  // An assignment tested for truth, while ((p = f())), narrows p.
  if (const auto *AssignBO = dyn_cast<BinaryOperator>(E)) {
    if (AssignBO->getOpcode() == BO_Assign)
      E = AssignBO->getLHS()->IgnoreParenImpCasts();
  }

  if (auto R = smartPtrGetRef(E)) {
    Results.push_back({std::move(*R), Negated});
    return;
  }

  if (auto R = PtrRef::fromExpr(E)) {
    if (R->getType()->isPointerType()) {
      Results.push_back({std::move(*R), Negated});
      return;
    }
    // A guard flag: if (valid) where valid = (p != nullptr), in a variable
    // or a field (if (d.valid)). Integer flags count too (int ok = p != NULL
    // is the C idiom).
    if (BoolGuards && R->getType()->isIntegerType()) {
      auto It = BoolGuards->find(*R);
      if (It != BoolGuards->end()) {
        // An outer ! flips the sense of every stored fact.
        for (ConditionResult CR : It->second) {
          CR.Negated = CR.Negated != Negated;
          Results.push_back(std::move(CR));
        }
        return;
      }
    }
  }

  if (const auto *MCE = dyn_cast<CXXMemberCallExpr>(E))
    analyzeSmartPtrBoolConversion(MCE, Negated, Results);
}

/// Extract the null-check facts a guard variable's initializer or assigned
/// value encodes, so a later if (flag) narrows the pointers it tested. Handles
/// p != nullptr, the ternary spellings p ? true : false / p ? 0 : 1, a
/// ternary with one constant false arm (p ? p->n : 0, p == nullptr ? false :
/// X), p && q (guard true means every conjunct held, so only true-direction
/// facts survive), and copies of other guards (bool c = !b).
static void computeGuardFacts(const Expr *Init, ASTContext &Ctx,
                              const NullState::BoolGuardMap &Guards,
                              SmallVectorImpl<ConditionResult> &Facts) {
  Init = Init->IgnoreParenImpCasts();
  if (const auto *ILE = dyn_cast<InitListExpr>(Init)) {
    if (ILE->getNumInits() != 1)
      return;
    Init = ILE->getInit(0)->IgnoreParenImpCasts();
  }
  if (const auto *CO = dyn_cast<AbstractConditionalOperator>(Init)) {
    std::optional<bool> TV = constantTruth(CO->getTrueExpr(), Ctx);
    std::optional<bool> FV = constantTruth(CO->getFalseExpr(), Ctx);
    if (TV && FV) {
      if (*TV == *FV)
        return;
      analyzeCondition(CO->getCond(), Ctx, Facts, &Guards);
      if (!*TV)
        for (auto &CR : Facts)
          CR.Negated = !CR.Negated;
      return;
    }
    // One arm a constant false and the other not constant: c ? X : false
    // holds exactly when c && X does, and c ? false : X when !c && X does,
    // so the guard being true proves c's facts (or !c's) and X's. As for &&,
    // only guard-true facts are kept. A constant true arm (c ? true : X is
    // c || X) proves nothing when the guard is true.
    bool FalseFirst = TV && !*TV;
    if (!FalseFirst && !(FV && !*FV))
      return;
    auto Add = [&Facts](ConditionResult CR) {
      if (!llvm::is_contained(Facts, CR))
        Facts.push_back(std::move(CR));
    };
    // c true, with c = a && b, means both held; c false, with c = a || b,
    // means neither did.
    SmallVector<ConditionResult, 4> CondFacts;
    decomposeChain(CO->getCond(), FalseFirst ? BO_LOr : BO_LAnd, Ctx, CondFacts,
                   &Guards);
    for (ConditionResult &CR : CondFacts) {
      if (CR.Negated != FalseFirst)
        continue;
      CR.Negated = false;
      Add(std::move(CR));
    }
    SmallVector<ConditionResult, 4> ArmFacts;
    computeGuardFacts(FalseFirst ? CO->getFalseExpr() : CO->getTrueExpr(), Ctx,
                      Guards, ArmFacts);
    for (ConditionResult &CR : ArmFacts)
      if (!CR.Negated)
        Add(std::move(CR));
    return;
  }
  if (const auto *BO = dyn_cast<BinaryOperator>(Init)) {
    if (BO->getOpcode() == BO_LAnd) {
      decomposeChain(BO, BO_LAnd, Ctx, Facts, &Guards);
      llvm::erase_if(Facts,
                     [](const ConditionResult &CR) { return CR.Negated; });
      return;
    }
  }
  analyzeCondition(Init, Ctx, Facts, &Guards);
}

/// Narrow a member path, plus every local copied from it
/// (T *q = s->next; if (s->next) *q;).
static void narrowMemberPath(NullState &NS, const MemberAccessPath &Path) {
  NS.markNarrowed(Path);
  for (const auto &[AliasVD, AliasPath] : NS.MemberAliases)
    if (AliasPath == Path)
      NS.markNarrowed(AliasVD);
}

/// Narrow a variable and everything known to hold the same pointer: its
/// alias target and all vars sharing that canonical target (y = x; z = x;
/// if (z) narrows z, x, and y), and the member path it was copied from
/// (T *q = s->next; if (q) *s->next;).
static void narrowVarWithAliases(NullState &NS, const VarDecl *VD) {
  NS.markNarrowed(VD);
  const VarDecl *Target = VD;
  auto AliasIt = NS.Aliases.find(VD);
  if (AliasIt != NS.Aliases.end()) {
    Target = AliasIt->second;
    NS.markNarrowed(Target);
  }
  for (const auto &[AliasVD, AliasTarget] : NS.Aliases)
    if (AliasTarget == VD || AliasTarget == Target)
      NS.markNarrowed(AliasVD);
  auto MemberIt = NS.MemberAliases.find(VD);
  if (MemberIt != NS.MemberAliases.end())
    narrowMemberPath(NS, MemberIt->second);
}

/// Apply one condition fact to a state (the caller picks the edge state
/// matching CR.Negated).
static void applyNarrowing(NullState &NS, const ConditionResult &CR) {
  SmallVector<PtrRef, 4> Work{CR.Ref};
  llvm::SmallPtrSet<const VarDecl *, 4> Seen;
  while (!Work.empty()) {
    PtrRef R = Work.pop_back_val();
    if (R.Path) {
      narrowMemberPath(NS, *R.Path);
      continue;
    }
    if (!R.VD)
      continue;
    narrowVarWithAliases(NS, R.VD);
    if (!R.VD->getType()->isPointerType() || !Seen.insert(R.VD).second)
      continue;
    auto It = NS.BoolGuards.find(PtrRef::var(R.VD));
    if (It == NS.BoolGuards.end())
      continue;
    for (const ConditionResult &Implied : It->second)
      if (!Implied.Negated)
        Work.push_back(Implied.Ref);
  }
}

//===----------------------------------------------------------------------===//
// Transfer functions
//===----------------------------------------------------------------------===//

/// Unwrap explicit casts and pointer arithmetic to the original pointer
/// expression. Template instantiations can bake _Nullable into a cast's
/// result type even when the source is unannotated, so when FoundCast is
/// set callers must judge nullability on the source type, not the cast.
/// Stops at a dynamic_cast, which can yield null from a non-null source: the
/// cast itself is then the origin.
static const Expr *unwrapCastsAndArithmetic(const Expr *E, bool &FoundCast) {
  FoundCast = false;
  while (true) {
    if (const auto *CE = dyn_cast<ExplicitCastExpr>(E)) {
      FoundCast = true;
      if (isa<CXXDynamicCastExpr>(CE))
        break;
      E = CE->getSubExpr()->IgnoreParenImpCasts();
    } else if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
      if (BO->getOpcode() == BO_Add || BO->getOpcode() == BO_Sub) {
        E = BO->getLHS()->getType()->isPointerType()
                ? BO->getLHS()->IgnoreParenImpCasts()
                : BO->getRHS()->IgnoreParenImpCasts();
      } else {
        break;
      }
    } else if (const auto *UO = dyn_cast<UnaryOperator>(E)) {
      // *p++ / *++p: the value dereferenced is p (or p+1), so its
      // null-ness is p's.
      if (UO->isIncrementDecrementOp())
        E = UO->getSubExpr()->IgnoreParenImpCasts();
      else
        break;
    } else {
      break;
    }
  }
  return E;
}

namespace {

/// Aggregate of every pointer-valued return seen during one function's
/// fixpoint; feeds the all-returns-nonnull summary emitted afterwards.
struct ReturnSummary {
  bool HasPointerReturn = false;
  bool AllNonnull = true;
};

/// What a pointer store writes, as far as the analysis can tell.
enum class StoredValue { Nonnull, Nullable, Unknown };

/// What a smart pointer holds after sp.reset(arg), judged from arg.
enum class ResetNullability { Null, Nonnull, Unknown };

/// Transfer functions for the dataflow analysis. Each CFG statement updates
/// the NullState: null checks narrow, assignments invalidate, and dereferences
/// of nullable pointers are reported through the Handler.
class TransferFunctions : public ConstStmtVisitor<TransferFunctions> {
  NullState &State;
  NullabilitySafetyHandler &Handler;
  ReturnSummary &Returns;
  ASTContext &Ctx;
  const NullabilitySafetyOptions &Options;
  const NullabilitySafetySummaries *Summaries;
  // The enclosing function declaration, needed for return type checking.
  const FunctionDecl *EnclosingFunc;
  // Function-scoped parent map; see isStdMoveInsideSmartPtrTransferCtx.
  const ParentMap *ParentMapPtr;
  // False during the fixpoint, whose intermediate states are not final:
  // reports and the return summary come only from the reporting pass over
  // converged block entry states. Summary queries run in both.
  bool Reporting;
  mutable bool StrictNonnull = false;

public:
  TransferFunctions(NullState &State, NullabilitySafetyHandler &Handler,
                    ReturnSummary &Returns, ASTContext &Ctx,
                    const NullabilitySafetyOptions &Options,
                    const NullabilitySafetySummaries *Summaries,
                    const FunctionDecl *EnclosingFunc, const ParentMap *PM,
                    bool Reporting)
      : State(State), Handler(Handler), Returns(Returns), Ctx(Ctx),
        Options(Options), Summaries(Summaries), EnclosingFunc(EnclosingFunc),
        ParentMapPtr(PM), Reporting(Reporting) {}

  bool isKnownAllReturnsNonnull(const FunctionDecl *Callee) const {
    return Summaries && Summaries->isKnownAllReturnsNonnull(Callee);
  }

  /// Classify each declared pointer, smart pointer, or guard flag from its
  /// initializer.
  void VisitDeclStmt(const DeclStmt *DS) {
    for (const auto *D : DS->decls()) {
      if (const auto *VD = dyn_cast<VarDecl>(D)) {
        // Each run of a declaration creates a new object (or binds a
        // reference anew), so facts about the variable from an earlier loop
        // iteration are stale: a pointer moved from or nulled at the end of
        // the body is not null when the next iteration declares it again. A
        // static local keeps its value, and its declaration runs once.
        if (VD->hasLocalStorage())
          forgetVariable(VD);
        if (VD->getType()->isPointerType()) {
          handlePointerVarInit(VD);
          continue;
        }
        handleSmartPtrVarInit(VD);
        recordGuardInit(VD);
      }
    }
  }

  /// Pointer arithmetic checks, then assignment handling by LHS shape.
  void VisitBinaryOperator(const BinaryOperator *BO) {
    checkPointerArithmetic(BO);

    if (!BO->isAssignmentOp())
      return;
    const Expr *LHS = BO->getLHS()->IgnoreParenImpCasts();

    handleStoreThroughPointer(LHS);

    if (auto LhsPath = decomposeMemberAccess(LHS))
      handleMemberAssign(BO, *LhsPath);

    if (const auto *DRE = dyn_cast<DeclRefExpr>(LHS))
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
        handleVarAssign(BO, VD);
  }

  /// *p dereference checks and p++ / p-- arithmetic checks.
  void VisitUnaryOperator(const UnaryOperator *UO) {
    if (UO->getOpcode() == UO_Deref) {
      const Expr *Operand = UO->getSubExpr()->IgnoreParenImpCasts();
      ExplicitCastInfo Cast = getExplicitCastInfo(Operand);
      if (!Cast.isNonnull()) {
        const Expr *SubExpr = lookThroughPtrToPtrCasts(Operand);

        if (const auto *DRE = dyn_cast<DeclRefExpr>(SubExpr)) {
          if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
            if (!VD->isImplicit() && !isNarrowed(VD))
              checkVarDeref(UO, VD);
          }
        } else if (const auto *ME = dyn_cast<MemberExpr>(SubExpr)) {
          checkMemberExprDeref(UO, ME);
        } else if (!isa<CXXThisExpr>(SubExpr)) {
          checkExprDeref(UO, SubExpr);
        }
      }
    }

    // p++ / --p: arithmetic on a nullable pointer warns, and the pointer now
    // points elsewhere so every fact about its old value is dropped.
    if (UO->getOpcode() == UO_PostInc || UO->getOpcode() == UO_PreInc ||
        UO->getOpcode() == UO_PostDec || UO->getOpcode() == UO_PreDec) {
      const Expr *SubExpr = UO->getSubExpr()->IgnoreParenImpCasts();
      if (const auto *DRE = dyn_cast<DeclRefExpr>(SubExpr)) {
        if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
          if (VD->getType()->isPointerType()) {
            checkVarArithmetic(UO, VD);
            forgetFactsAbout(VD);
          } else if (VD->getType()->isIntegerType()) {
            // A guard flag that changes value no longer encodes the check.
            State.BoolGuards.erase(PtrRef::var(VD));
          }
        }
      } else if (auto Path = decomposeMemberAccess(SubExpr)) {
        if (Path->leafField()->getType()->isIntegerType())
          State.BoolGuards.erase(PtrRef{nullptr, std::move(*Path)});
      }
    }
  }

  /// p->field dereference checks, including smart pointer operator->.
  void VisitMemberExpr(const MemberExpr *ME) {
    if (!ME->isArrow())
      return;

    const Expr *Operand = ME->getBase()->IgnoreParenImpCasts();
    ExplicitCastInfo Cast = getExplicitCastInfo(Operand);
    if (Cast.isNonnull())
      return;

    const Expr *Base = lookThroughPtrToPtrCasts(Operand);

    if (isa<CXXThisExpr>(Base))
      return;

    if (checkSmartPtrArrow(ME, Base))
      return;

    if (const auto *DRE = dyn_cast<DeclRefExpr>(Base)) {
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        if (!isNarrowed(VD))
          checkVarDeref(ME, VD);
      }
    } else if (const auto *BaseME = dyn_cast<MemberExpr>(Base)) {
      checkMemberExprDeref(ME, BaseME);
    } else {
      checkExprDeref(ME, Base);
    }
  }

  /// p[i] dereference checks. An array-typed base or (&x)[i] cannot be null.
  void VisitArraySubscriptExpr(const ArraySubscriptExpr *ASE) {
    const Expr *Operand = ASE->getBase()->IgnoreParenImpCasts();
    ExplicitCastInfo Cast = getExplicitCastInfo(Operand);
    if (Cast.isNonnull())
      return;

    const Expr *Base = lookThroughPtrToPtrCasts(Operand);
    if (const auto *UO = dyn_cast<UnaryOperator>(Base))
      if (UO->getOpcode() == UO_AddrOf)
        return;
    if (const auto *DRE = dyn_cast<DeclRefExpr>(Base)) {
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        if (!isNarrowed(VD) && !VD->getType()->isArrayType())
          checkVarDeref(ASE, VD);
      }
    } else if (const auto *ME = dyn_cast<MemberExpr>(Base)) {
      if (!ME->getType()->isArrayType())
        checkMemberExprDeref(ASE, ME);
    } else {
      QualType BaseTy = Base->getType();
      if (!BaseTy->isArrayType())
        checkExprDeref(ASE, Base);
    }
  }

  /// Calls never invalidate narrowing; see the file overview for why.
  void VisitCallExpr(const CallExpr *CE) {
    if (const auto *Callee = CE->getDirectCallee()) {
      narrowFromBuiltinAssume(CE, Callee);
      checkCallArguments(CE, Callee);
    }

    if (const auto *MCE = dyn_cast<CXXMemberCallExpr>(CE)) {
      handleSmartPtrReset(MCE);
      handleSmartPtrReleaseOrSwap(MCE);
    }
    handleStdSwap(CE);

    if (const auto *OCE = dyn_cast<CXXOperatorCallExpr>(CE)) {
      handleSmartPtrAssign(OCE);
      // *sp through operator* is checked like operator->.
      if (OCE->getOperator() == OO_Star && OCE->getNumArgs() >= 1) {
        const Expr *Obj = OCE->getArg(0);
        if (isSmartPointerObject(Obj))
          checkSmartPtrDeref(CE, Obj);
      }
    }

    handleBareStdMove(CE);
  }

  /// Check constructor arguments against parameter nullability, same as
  /// VisitCallExpr does for regular function calls.
  void VisitCXXConstructExpr(const CXXConstructExpr *CE) {
    const CXXConstructorDecl *Ctor = CE->getConstructor();
    if (!Ctor)
      return;
    const auto *NNAttr = Ctor->getAttr<NonNullAttr>();
    for (unsigned I = 0, N = std::min(CE->getNumArgs(), Ctor->getNumParams());
         I < N; ++I) {
      const ParmVarDecl *Param = Ctor->getParamDecl(I);
      if (!Param->getType()->isPointerType())
        continue;
      bool ParamIsNonnull =
          isNonnullType(Param->getType()) || (NNAttr && NNAttr->isNonNull(I));
      if (ParamIsNonnull)
        checkNonnullParamArg(CE->getArg(I), Param);
    }
  }

  /// Aggregate init lists: evidence for every pointer field (never Nonnull,
  /// so an aggregate alone cannot make a field a candidate), and a warning
  /// for a null value in a _Nonnull field (S{nullptr, &x}).
  void VisitInitListExpr(const InitListExpr *ILE) {
    const auto *RT = ILE->getType()->getAs<RecordType>();
    if (!RT)
      return;
    const RecordDecl *RD = RT->getDecl();
    if (!RD || (!RD->isStruct() && !RD->isClass()))
      return;
    auto FI = RD->field_begin();
    for (unsigned I = 0, N = ILE->getNumInits(); I < N && FI != RD->field_end();
         ++I, ++FI) {
      const FieldDecl *FD = *FI;
      if (!FD->getType()->isPointerType())
        continue;
      const Expr *Init = ILE->getInit(I);
      if (Reporting) {
        NullabilityEvidence Kind =
            classifyEvidence(Init, isProvablyNonnull(Init));
        if (Kind != NullabilityEvidence::Nonnull)
          Handler.handleMemberAssignEvidence(ILE, FD, Kind);
      }
      if (!isNonnullType(FD->getType()))
        continue;
      if (classifyStoredValue(Init) == StoredValue::Nullable) {
        if (Reporting) {
          ++NumAssignmentWarnings;
          Handler.handleNullableMemberAssignment(Init, FD);
        }
      }
    }
  }

  /// Return evidence, the all-returns-nonnull summary, and the nullable
  /// return from a _Nonnull function.
  void VisitReturnStmt(const ReturnStmt *RS) {
    if (!EnclosingFunc)
      return;
    const Expr *RetVal = RS->getRetValue();
    if (!RetVal)
      return;
    QualType RetType = EnclosingFunc->getReturnType();
    if (!RetType->isPointerType())
      return;

    bool RetIsNonnull = !isExprNullable(RetVal);
    if (!Reporting)
      return;
    bool RetIsProvablyNonnull = isProvablyNonnull(RetVal);
    Returns.HasPointerReturn = true;
    Returns.AllNonnull &= RetIsProvablyNonnull;
    // Return evidence is skipped for lambdas and other non-identifier-named
    // functions: they have no cross-TU identity, and getName() would assert.
    if (EnclosingFunc->getDeclName().isIdentifier()) {
      Handler.handleReturnEvidence(
          RetVal, EnclosingFunc,
          classifyEvidence(RetVal, RetIsProvablyNonnull));
    }

    if (isNonnullType(RetType) && !RetIsNonnull) {
      ++NumReturnWarnings;
      Handler.handleNullableReturn(RetVal);
    }
  }

  /// Evidence for a constructor's member initializers (': field(expr)'),
  /// judged like member assignments against the entry state. They are
  /// CXXCtorInitializers, not BinaryOperators, so VisitBinaryOperator never
  /// sees them.
  void reportCtorInitEvidence(const CXXConstructorDecl *CD) {
    assert(Reporting && "evidence is emitted only by the reporting pass");
    for (const auto *CI : CD->inits()) {
      if (!CI->isAnyMemberInitializer())
        continue;
      const FieldDecl *FD = CI->getMember();
      const Expr *Init = CI->getInit();
      if (!FD || !Init || !FD->getType()->isPointerType())
        continue;
      StoredValue V = classifyStoredValue(Init);
      bool IsNonnull =
          V == StoredValue::Nonnull ||
          (V == StoredValue::Unknown && isNonnullType(FD->getType()));
      Handler.handleMemberAssignEvidence(Init->IgnoreParenImpCasts(), FD,
                                         classifyEvidence(Init, IsNonnull));
    }
  }

private:
  /// Classify a raw pointer variable from its initializer: record aliases,
  /// then store the initial value. An uninitialized _Nonnull local is trusted
  /// to be non-null.
  void handlePointerVarInit(const VarDecl *VD) {
    if (!VD->hasInit()) {
      if (isNonnullType(VD->getType()))
        State.markNarrowed(VD);
      return;
    }
    recordPointerSource(VD, VD->getInit());
    recordTernaryPointerGuard(VD, VD->getInit());
    recordPointerImplication(VD, VD->getInit());
    storeToVar(VD, VD->getInit(), VD->getInit());
  }

  /// Whether a smart pointer initializer may leave it null: nullptr, default
  /// construction, a _Nullable-returning call, or std::dynamic_pointer_cast,
  /// which yields null when the runtime check fails whatever its argument
  /// (like a raw dynamic_cast, whatever the default).
  bool isNullSmartPtrInit(const Expr *Init) const {
    if (Init->isNullPointerConstant(Ctx, Expr::NPC_ValueDependentIsNotNull))
      return true;
    if (const Expr *Arg = getNullPreservingPointerCastArg(Init))
      return isNullSmartPtrInit(unwrapImplicitWrappers(Arg));
    if (const auto *CCE = dyn_cast<CXXConstructExpr>(Init)) {
      if (CCE->getNumArgs() == 0)
        return CCE->getConstructor()->isDefaultConstructor();
      if (CCE->getNumArgs() == 1)
        return isNullSmartPtrInit(unwrapImplicitWrappers(CCE->getArg(0)));
      return false;
    }
    if (const auto *CE = dyn_cast<CallExpr>(Init))
      return isExplicitlyNullableType(CE->getCallReturnType(Ctx)) ||
             isStdCallTo(CE, "dynamic_pointer_cast");
    return false;
  }

  /// Track smart pointer initialization: narrow if constructed from a
  /// provably non-null source (make_unique, make_shared, new) or copied from
  /// a smart pointer that is non-null right now, taint if null, or inherit
  /// the source's state from auto x = std::move(other). A copy owns its own
  /// reference, so it is not linked to the source afterwards.
  void handleSmartPtrVarInit(const VarDecl *VD) {
    // A reference bound to a variable is that variable (see
    // resolveSmartPtrReference): binding it neither copies nor moves.
    if (resolveSmartPtrReference(VD) != VD)
      return;
    // Strip reference: range-for loop variables have type const T&.
    if (isSmartPointerType(VD->getType().getNonReferenceType()) &&
        VD->hasInit()) {
      const Expr *Init = unwrapImplicitWrappers(VD->getInit());
      if (isNonnullSmartPtrInit(Init) ||
          isInitFromNonnullContainerElement(VD)) {
        State.markNarrowed(VD);
      } else if (isNullSmartPtrInit(Init)) {
        State.markNullable(VD);
      } else if (const Expr *Src = smartPtrCopySource(Init)) {
        // A copy holds what the source holds. A reference binds to the
        // source itself rather than copying it, so a later reset() of the
        // source must reach it.
        if (!VD->getType()->isReferenceType()) {
          if (!isSmartPointerMaybeNull(Src))
            State.markNarrowed(VD);
          else if (isSmartPointerKnownNullable(Src))
            State.markNullable(VD);
        }
      } else {
        // auto x = std::move(other); inherits the source's state. The
        // standalone std::move handler skipped the source erase (see
        // isStdMoveInsideSmartPtrTransferCtx), so the source's pre-move
        // state is still there.
        const Expr *Inner = lookThroughNullPreservingConversions(Init);
        if (const auto *CE = dyn_cast<CallExpr>(Inner)) {
          if (CE->isCallToStdMove() && CE->getNumArgs() >= 1 &&
              !VD->getType()->isReferenceType()) {
            if (auto Src = smartPtrRef(CE->getArg(0))) {
              if (State.isNarrowed(*Src))
                State.markNarrowed(VD);
              else if (isSmartPointerKnownNullable(CE->getArg(0)))
                State.markNullable(VD);
              forgetSmartPtrFacts(*Src);
              State.markNullable(*Src);
            }
          }
        }
      }
    }
  }

  /// Record the null-check facts a guard's initializer encodes, so that after
  /// bool valid = (p != nullptr) or int ok = p ? 1 : 0, testing the guard
  /// narrows p.
  void recordGuardInit(const VarDecl *VD) {
    if (VD->getType()->isIntegerType() && VD->hasInit()) {
      SmallVector<ConditionResult, 2> Facts;
      computeGuardFacts(VD->getInit(), Ctx, State.BoolGuards, Facts);
      if (!Facts.empty())
        State.BoolGuards[PtrRef::var(VD)] = std::move(Facts);
    }
  }

  /// The inverse of recordGuardInit. A pointer initialized from a ternary
  /// with one provably non-null arm and one null constant is non-null
  /// exactly when the condition holds, so p = c ? &x : nullptr records
  /// {c -> [(p, Negated=false)]} and a later if (c) narrows p through the
  /// existing guard-variable path. The condition must be a plain integer
  /// variable (optionally negated); anything else has no VarDecl to key on.
  void recordTernaryPointerGuard(const VarDecl *PtrVD, const Expr *Init) {
    if (!Init || !PtrVD->getType()->isPointerType())
      return;
    const auto *CO =
        dyn_cast<AbstractConditionalOperator>(Init->IgnoreParenImpCasts());
    if (!CO)
      return;
    const Expr *TrueArm = CO->getTrueExpr()->IgnoreParenImpCasts();
    const Expr *FalseArm = CO->getFalseExpr()->IgnoreParenImpCasts();
    bool TrueIsNull = TrueArm->IgnoreParenCasts()->isNullPointerConstant(
        Ctx, Expr::NPC_ValueDependentIsNotNull);
    bool FalseIsNull = FalseArm->IgnoreParenCasts()->isNullPointerConstant(
        Ctx, Expr::NPC_ValueDependentIsNotNull);
    if (TrueIsNull == FalseIsNull)
      return;
    const Expr *NonnullArm = TrueIsNull ? FalseArm : TrueArm;
    if (!isNonnullInit(NonnullArm) && !isExprNarrowedNonnull(NonnullArm) &&
        !isNonnullType(NonnullArm->getType()))
      return;
    bool Negated = TrueIsNull;
    const Expr *Cond = CO->getCond()->IgnoreParenImpCasts();
    while (const auto *UO = dyn_cast<UnaryOperator>(Cond)) {
      if (UO->getOpcode() != UO_LNot)
        break;
      Negated = !Negated;
      Cond = UO->getSubExpr()->IgnoreParenImpCasts();
    }
    const auto *DRE = dyn_cast<DeclRefExpr>(Cond);
    if (!DRE)
      return;
    const auto *GuardVD = dyn_cast<VarDecl>(DRE->getDecl());
    if (!GuardVD || !GuardVD->getType()->isIntegerType())
      return;
    ExprMutationAnalyzer MutationAnalyzer(*CO, Ctx);
    if (MutationAnalyzer.isMutated(GuardVD))
      return;
    ConditionResult CR{PtrRef{PtrVD, std::nullopt}, Negated};
    auto &Facts = State.BoolGuards[PtrRef::var(GuardVD)];
    // Re-recording on each fixpoint iteration must not grow the vector, or
    // the block entry state never compares equal and never converges.
    if (!llvm::is_contained(Facts, CR))
      Facts.push_back(CR);
  }

  void recordPointerImplication(const VarDecl *PtrVD, const Expr *Init) {
    if (!Init || !PtrVD->getType()->isPointerType())
      return;
    State.BoolGuards.erase(PtrRef::var(PtrVD));
    const auto *CO =
        dyn_cast<AbstractConditionalOperator>(Init->IgnoreParenImpCasts());
    if (!CO)
      return;
    auto IsNullArm = [&](const Expr *Arm) {
      return Arm->IgnoreParenCasts()->isNullPointerConstant(
          Ctx, Expr::NPC_ValueDependentIsNotNull);
    };
    bool TrueIsNull = IsNullArm(CO->getTrueExpr());
    if (TrueIsNull == IsNullArm(CO->getFalseExpr()))
      return;
    SmallVector<ConditionResult, 4> Facts;
    decomposeChain(CO->getCond(), TrueIsNull ? BO_LOr : BO_LAnd, Ctx, Facts,
                   &State.BoolGuards);
    ExprMutationAnalyzer MutationAnalyzer(*CO, Ctx);
    SmallVector<ConditionResult, 2> Implied;
    for (ConditionResult CR : Facts) {
      if (CR.Negated != TrueIsNull)
        continue;
      const VarDecl *Root = CR.Ref.VD ? CR.Ref.VD : CR.Ref.Path->Root;
      if (Root == PtrVD || (Root && MutationAnalyzer.isMutated(Root)))
        continue;
      CR.Negated = false;
      if (!llvm::is_contained(Implied, CR))
        Implied.push_back(std::move(CR));
    }
    if (!Implied.empty())
      State.BoolGuards[PtrRef::var(PtrVD)] = std::move(Implied);
  }

  /// __builtin_assume(cond) narrows pointers mentioned in cond.
  void narrowFromBuiltinAssume(const CallExpr *CE, const FunctionDecl *Callee) {
    if (Callee->getBuiltinID() == Builtin::BI__builtin_assume &&
        CE->getNumArgs() >= 1) {
      const Expr *Arg = CE->getArg(0)->IgnoreParenImpCasts();
      SmallVector<ConditionResult, 2> Results;
      analyzeCondition(Arg, Ctx, Results, &State.BoolGuards);
      for (const auto &CR : Results)
        if (!CR.Negated)
          applyNarrowing(State, CR);
    }
  }

  /// Check each argument against its parameter's nullability, then emit
  /// parameter evidence (including nullptr default arguments).
  void checkCallArguments(const CallExpr *CE, const FunctionDecl *Callee) {
    // Narrow pointers passed to _Nonnull or __attribute__((nonnull))
    // parameters: surviving the call proves them non-null. For member
    // operator calls (e.g. lambda operator()) getArg(0) is the implicit
    // object, so real args start at offset 1.
    unsigned ArgOffset = 0;
    if (isa<CXXOperatorCallExpr>(CE) && isa<CXXMethodDecl>(Callee))
      ArgOffset = 1;
    const auto *NNAttr = Callee->getAttr<NonNullAttr>();
    unsigned EffArgs = CE->getNumArgs() - ArgOffset;
    // Parameter evidence for cross-TU inference is skipped for builtins,
    // empty-named functions, and lambda operator() calls (lambda params
    // have no cross-TU identity).
    bool IsLambdaCall = false;
    if (const auto *MD = dyn_cast<CXXMethodDecl>(Callee))
      IsLambdaCall = MD->getParent()->isLambda();
    bool EmitEvidence = !Callee->getBuiltinID() &&
                        !Callee->getDeclName().isEmpty() && !IsLambdaCall;
    for (unsigned I = 0, N = std::min(EffArgs, Callee->getNumParams()); I < N;
         ++I) {
      const ParmVarDecl *Param = Callee->getParamDecl(I);
      if (!Param->getType()->isPointerType())
        continue;
      bool ParamIsNonnull =
          isNonnullType(Param->getType()) || (NNAttr && NNAttr->isNonNull(I));
      // Lambda pointer parameters are treated as non-null inside the body, so
      // check the call site instead: passing a nullable value to a lambda
      // parameter that is not explicitly _Nullable warns.
      if (!ParamIsNonnull && IsLambdaCall &&
          Options.DefaultNullability != NullabilityKind::NonNull &&
          !isExplicitlyNullableType(Param->getType()))
        ParamIsNonnull = true;
      if (ParamIsNonnull)
        checkNonnullParamArg(CE->getArg(I + ArgOffset), Param);
    }

    for (unsigned I = 0, N = std::min(EffArgs, Callee->getNumParams()); I < N;
         ++I)
      if (const VarDecl *VD = pointerWritableByCallee(
              CE->getArg(I + ArgOffset), Callee->getParamDecl(I)->getType()))
        escapeToCallee(VD);

    // Evidence comes last so every argument is judged after all
    // nonnull-parameter narrowing from this call has been applied.
    if (EmitEvidence && Reporting) {
      for (unsigned I = 0, N = std::min(EffArgs, Callee->getNumParams()); I < N;
           ++I) {
        const ParmVarDecl *Param = Callee->getParamDecl(I);
        if (!Param->getType()->isPointerType())
          continue;
        const Expr *Arg = CE->getArg(I + ArgOffset)->IgnoreParenImpCasts();
        Handler.handleParameterEvidence(
            CE->getArg(I + ArgOffset), Param, Callee,
            classifyEvidence(Arg, isProvablyNonnull(Arg)));
      }
      // Parameters with nullptr default arguments are nullable evidence
      // even when callers always pass nonnull explicitly: the function
      // can be called without that argument, receiving nullptr.
      for (unsigned I = 0, N = Callee->getNumParams(); I < N; ++I) {
        const ParmVarDecl *Param = Callee->getParamDecl(I);
        if (!Param->getType()->isPointerType() || !Param->hasDefaultArg())
          continue;
        if (Param->hasUninstantiatedDefaultArg())
          continue;
        const Expr *DefArg = Param->getDefaultArg();
        if (DefArg && DefArg->isNullPointerConstant(
                          Ctx, Expr::NPC_ValueDependentIsNotNull))
          Handler.handleParameterEvidence(DefArg, Param, Callee,
                                          NullabilityEvidence::Nullable);
      }
    }
  }

  const VarDecl *pointerWritableByCallee(const Expr *Arg,
                                         QualType ParamTy) const {
    Arg = Arg->IgnoreParenImpCasts();
    auto pointerVar = [](const Expr *E) -> const VarDecl * {
      if (const auto *DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts()))
        if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
          if (VD->getType()->isPointerType())
            return VD;
      return nullptr;
    };
    if (const auto *RT = ParamTy->getAs<LValueReferenceType>()) {
      QualType Referee = RT->getPointeeType();
      if (Referee->isPointerType() && !Referee.isConstQualified())
        return pointerVar(Arg);
      return nullptr;
    }
    const auto *PT = ParamTy->getAs<PointerType>();
    if (!PT || !PT->getPointeeType()->isPointerType() ||
        PT->getPointeeType().isConstQualified())
      return nullptr;
    if (const auto *UO = dyn_cast<UnaryOperator>(Arg))
      if (UO->getOpcode() == UO_AddrOf)
        return pointerVar(UO->getSubExpr());
    if (const VarDecl *PPVD = pointerVar(Arg)) {
      auto It = State.AddrOfTargets.find(PPVD);
      if (It != State.AddrOfTargets.end())
        return It->second;
    }
    return nullptr;
  }

  void escapeToCallee(const VarDecl *VD) {
    State.NullableVars.erase(VD);
    State.NullableMembers.remove_if(
        [VD](const MemberAccessPath &Path) { return Path.Root == VD; });
    invalidateBoolGuardsFor(VD);
    State.BoolGuards.erase(PtrRef::var(VD));
    State.AddrOfTargets.erase(VD);
  }

  /// Handle sp.reset() / sp.reset(ptr), a CXXMemberCallExpr.
  void handleSmartPtrReset(const CXXMemberCallExpr *MCE) {
    const Expr *Obj = MCE->getImplicitObjectArgument();
    if (isSmartPointerObject(Obj)) {
      if (const auto *MD = MCE->getMethodDecl()) {
        if (MD->getDeclName().isIdentifier() && MD->getName() == "reset") {
          // reset(nullptr) makes it null; reset(proven_nonnull) narrows it;
          // an unknown argument clears both facts. libc++ declares
          // reset(pointer p = pointer()), so a no-arg call arrives with a
          // CXXDefaultArgExpr arg, which means reset to null however the
          // default is spelled.
          ResetNullability Result = ResetNullability::Null;
          if (MCE->getNumArgs() > 0) {
            const Expr *Arg = MCE->getArg(0);
            if (!isa<CXXDefaultArgExpr>(Arg)) {
              Arg = Arg->IgnoreParenImpCasts();
              if (Arg->isNullPointerConstant(Ctx,
                                             Expr::NPC_ValueDependentIsNotNull))
                Result = ResetNullability::Null;
              else if (isNonnullInit(Arg))
                Result = ResetNullability::Nonnull;
              else
                Result = ResetNullability::Unknown;
            }
          }
          if (auto R = smartPtrRef(Obj)) {
            forgetSmartPtrFacts(*R);
            State.clear(*R);
            if (Result == ResetNullability::Nonnull)
              State.markNarrowed(*R);
            else if (Result == ResetNullability::Null)
              State.markNullable(*R);
          }
        }
      }
    }
  }

  void handleSmartPtrReleaseOrSwap(const CXXMemberCallExpr *MCE) {
    const Expr *Obj = MCE->getImplicitObjectArgument();
    const auto *MD = MCE->getMethodDecl();
    if (!MD || !isSmartPointerObject(Obj) || !MD->getDeclName().isIdentifier())
      return;
    if (MD->getName() == "release") {
      if (auto R = smartPtrRef(Obj)) {
        forgetSmartPtrFacts(*R);
        State.clear(*R);
        State.markNullable(*R);
      }
    } else if (MD->getName() == "swap" && MCE->getNumArgs() == 1) {
      swapSmartPtrFacts(Obj, MCE->getArg(0));
    }
  }

  void handleStdSwap(const CallExpr *CE) {
    const FunctionDecl *Callee = CE->getDirectCallee();
    if (!Callee || CE->getNumArgs() != 2 || !Callee->isInStdNamespace() ||
        !Callee->getDeclName().isIdentifier() || Callee->getName() != "swap")
      return;
    if (isSmartPointerObject(CE->getArg(0)) &&
        isSmartPointerObject(CE->getArg(1)))
      swapSmartPtrFacts(CE->getArg(0), CE->getArg(1));
  }

  void swapSmartPtrFacts(const Expr *A, const Expr *B) {
    std::optional<PtrRef> RA = smartPtrRef(A), RB = smartPtrRef(B);
    bool ANarrowed = RA && State.isNarrowed(*RA);
    bool ANullable = RA && State.isNullable(*RA);
    bool BNarrowed = RB && State.isNarrowed(*RB);
    bool BNullable = RB && State.isNullable(*RB);
    if (RA)
      forgetSmartPtrFacts(*RA);
    if (RB)
      forgetSmartPtrFacts(*RB);
    if (RA) {
      State.clear(*RA);
      if (BNarrowed)
        State.markNarrowed(*RA);
      else if (BNullable)
        State.markNullable(*RA);
    }
    if (RB) {
      State.clear(*RB);
      if (ANarrowed)
        State.markNarrowed(*RB);
      else if (ANullable)
        State.markNullable(*RB);
    }
  }

  /// Handle sp = nullptr / sp = make_unique(...) / sp = std::move(other) /
  /// sp = other (a copy, non-null when other is non-null right now).
  /// LHS may be a local (VarDecl), a this-member, or a member access chain.
  void handleSmartPtrAssign(const CXXOperatorCallExpr *OCE) {
    if (OCE->getOperator() == OO_Equal && OCE->getNumArgs() >= 2) {
      const Expr *LhsArg = OCE->getArg(0);
      std::optional<PtrRef> Lhs = smartPtrRef(LhsArg);

      if (Lhs) {
        const Expr *RHS = unwrapImplicitWrappers(OCE->getArg(1));
        const Expr *CopySrc = smartPtrCopySource(RHS);
        // sp = sp changes nothing, like p = p on a raw pointer: its facts,
        // and the guards that tested it, still hold.
        if (CopySrc && smartPtrRef(CopySrc) == Lhs)
          return;
        // Judge a copy source before the LHS is cleared.
        bool CopyIsNonnull = CopySrc && !isSmartPointerMaybeNull(CopySrc);
        bool CopyIsNullable = CopySrc && isSmartPointerKnownNullable(CopySrc);
        forgetSmartPtrFacts(*Lhs);
        // Clear the LHS's facts. A value the checks below cannot classify
        // leaves a local unknown, like a raw pointer store. A member path is
        // marked nullable instead unless unknown smart pointers are trusted
        // (the nonnull default): a this-> member is trusted unless flow
        // marks it (see warnSmartPtrDeref), where a raw member would fall
        // back to its declared type.
        if (Lhs->VD || Options.DefaultNullability == NullabilityKind::NonNull)
          State.clear(*Lhs);
        else
          State.markNullable(*Lhs);

        if (isNonnullSmartPtrInit(RHS)) {
          State.markNarrowed(*Lhs);
        } else if (isNullSmartPtrInit(RHS)) {
          State.markNullable(*Lhs);
        } else if (const auto *RhsCE = dyn_cast<CallExpr>(
                       lookThroughNullPreservingConversions(RHS))) {
          if (RhsCE->isCallToStdMove() && RhsCE->getNumArgs() >= 1) {
            // sp = std::move(other), possibly through a null-preserving
            // cast: LHS inherits source's state.
            auto Src = smartPtrRef(RhsCE->getArg(0));
            bool SrcIsNullable =
                Src && isSmartPointerKnownNullable(RhsCE->getArg(0));
            if (Src)
              forgetSmartPtrFacts(*Src);
            if (Src && Src->VD) {
              if (isNarrowed(Src->VD))
                State.markNarrowed(*Lhs);
              else if (SrcIsNullable)
                State.markNullable(*Lhs);
              State.markNullable(Src->VD);
            }
          } else if (isNonnullType(RhsCE->getType())) {
            State.markNarrowed(*Lhs);
          }
        } else if (CopyIsNonnull) {
          State.markNarrowed(*Lhs);
        } else if (CopyIsNullable) {
          State.markNullable(*Lhs);
        }
      } else if (auto StructPath = decomposeMemberAccess(LhsArg)) {
        // Non-smart-pointer struct member assignment (e.g. o.inner = fresh):
        // invalidate any narrowed paths nested under the LHS.
        invalidateMembersWithPrefix(*StructPath);
      } else if (const auto *DRE =
                     dyn_cast<DeclRefExpr>(LhsArg->IgnoreParenImpCasts())) {
        // A struct variable assigned as a whole (d = other): the same for
        // every path rooted at it.
        if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
          invalidateMembersFor(VD);
      }
    }
  }

  /// A bare std::move(sp) (an argument, a cast operand, anything but a
  /// smart-pointer transfer) leaves the source moved-from, so it is nullable
  /// until reassigned or null-checked, even when declared _Nonnull. Inside a
  /// smart-pointer transfer (see isStdMoveInsideSmartPtrTransferCtx) the
  /// transfer handler owns the source taint instead; no parent map means "not
  /// a transfer".
  void handleBareStdMove(const CallExpr *CE) {
    if (CE->isCallToStdMove() && CE->getNumArgs() >= 1 &&
        (!ParentMapPtr ||
         !isStdMoveInsideSmartPtrTransferCtx(CE, *ParentMapPtr))) {
      if (auto R = smartPtrRef(CE->getArg(0))) {
        forgetSmartPtrFacts(*R);
        State.markNullable(*R);
      }
    }
  }

  /// Warn on arithmetic (p + i, p - i, p += i, p -= i) whose pointer operand
  /// may be null.
  void checkPointerArithmetic(const BinaryOperator *BO) {
    // p + 0 and p - 0 are identity operations and never warn.
    if (BO->getOpcode() == BO_Add || BO->getOpcode() == BO_Sub) {
      const Expr *PtrExpr = nullptr;
      const Expr *OtherExpr = nullptr;
      if (BO->getLHS()->getType()->isPointerType()) {
        PtrExpr = BO->getLHS()->IgnoreParenImpCasts();
        OtherExpr = BO->getRHS()->IgnoreParenImpCasts();
      } else if (BO->getRHS()->getType()->isPointerType()) {
        PtrExpr = BO->getRHS()->IgnoreParenImpCasts();
        OtherExpr = BO->getLHS()->IgnoreParenImpCasts();
      }
      if (PtrExpr) {
        bool IsZeroOffset = false;
        if (OtherExpr && !OtherExpr->getType()->isPointerType()) {
          if (auto Val = OtherExpr->getIntegerConstantExpr(Ctx))
            if (*Val == 0)
              IsZeroOffset = true;
        }
        if (!IsZeroOffset) {
          if (const auto *DRE = dyn_cast<DeclRefExpr>(PtrExpr))
            if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
              checkVarArithmetic(BO, VD);
          if (OtherExpr && OtherExpr->getType()->isPointerType())
            if (const auto *DRE = dyn_cast<DeclRefExpr>(OtherExpr))
              if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
                checkVarArithmetic(BO, VD);
        }
      }
    }

    if (BO->getOpcode() == BO_AddAssign || BO->getOpcode() == BO_SubAssign) {
      const Expr *LHS = BO->getLHS()->IgnoreParenImpCasts();
      if (LHS->getType()->isPointerType())
        if (const auto *DRE = dyn_cast<DeclRefExpr>(LHS))
          if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
            checkVarArithmetic(BO, VD);
    }
  }

  /// *pp = X where pp is known to hold &local: drop local's narrowing. Only a
  /// precisely known target is invalidated; guessing would flood downstream
  /// dereferences with false positives.
  void handleStoreThroughPointer(const Expr *LHS) {
    if (const auto *UO = dyn_cast<UnaryOperator>(LHS)) {
      if (UO->getOpcode() == UO_Deref) {
        const Expr *Sub = UO->getSubExpr()->IgnoreParenImpCasts();
        if (const auto *SubDRE = dyn_cast<DeclRefExpr>(Sub)) {
          if (const auto *PPVD = dyn_cast<VarDecl>(SubDRE->getDecl())) {
            auto It = State.AddrOfTargets.find(PPVD);
            if (It != State.AddrOfTargets.end()) {
              const VarDecl *TgtVD = It->second;
              // The target loses its narrowing and is not re-narrowed from
              // the RHS.
              State.NarrowedVars.erase(TgtVD);
              State.NullableVars.erase(TgtVD);
              invalidateMembersFor(TgtVD);
              invalidateBoolGuardsFor(TgtVD);
              State.BoolGuards.erase(PtrRef::var(TgtVD));
            }
          }
        }
      }
    }
  }

  /// Assignment to a member (this->field, var->field, s.field, or nested
  /// like s.inner.field) invalidates any narrowing on that member and below
  /// it, then stores the RHS.
  void handleMemberAssign(const BinaryOperator *BO,
                          const MemberAccessPath &LhsPath) {
    const FieldDecl *FD = LhsPath.leafField();
    invalidateMembersWithPrefix(LhsPath);
    if (BO->getOpcode() != BO_Assign)
      return;
    // A guard field (d.ok = p != nullptr) records its facts like a guard
    // variable.
    if (FD->getType()->isIntegerType()) {
      SmallVector<ConditionResult, 2> Facts;
      computeGuardFacts(BO->getRHS(), Ctx, State.BoolGuards, Facts);
      if (!Facts.empty())
        State.BoolGuards[PtrRef{nullptr, LhsPath}] = std::move(Facts);
      return;
    }
    if (!FD->getType()->isPointerType())
      return;
    bool Narrowed = storePointer(PtrRef{nullptr, LhsPath}, BO->getRHS(), BO);
    if (Reporting)
      Handler.handleMemberAssignEvidence(
          BO, FD, classifyEvidence(BO->getRHS(), Narrowed));
  }

  /// Assignment to a local or parameter: guard flags re-capture their
  /// facts, pointers lose their old facts and are re-classified from the
  /// RHS.
  void handleVarAssign(const BinaryOperator *BO, const VarDecl *VD) {
    // Guard reassignment replaces any stored facts with whatever the
    // new value encodes (ok = p != NULL), or nothing.
    if (VD->getType()->isIntegerType()) {
      State.BoolGuards.erase(PtrRef::var(VD));
      if (BO->getOpcode() == BO_Assign) {
        SmallVector<ConditionResult, 2> Facts;
        computeGuardFacts(BO->getRHS(), Ctx, State.BoolGuards, Facts);
        if (!Facts.empty())
          State.BoolGuards[PtrRef::var(VD)] = std::move(Facts);
      }
      return;
    }
    // A C struct assigned as a whole (d = other; C++ calls operator=, see
    // handleSmartPtrAssign): every fact about a member path under it is
    // stale.
    if (VD->getType()->isRecordType()) {
      invalidateMembersFor(VD);
      return;
    }
    if (!VD->getType()->isPointerType())
      return;
    // p += n / p -= n: arithmetic on a non-null pointer stays
    // non-null (and on a nullable one was already diagnosed above), so
    // keep the narrowed/nullable flags and only drop facts that named
    // the old value.
    if (BO->getOpcode() == BO_AddAssign || BO->getOpcode() == BO_SubAssign) {
      forgetFactsAbout(VD);
      return;
    }
    // Self-assignment (p = p, p = (T *)p) changes nothing.
    if (BO->getOpcode() == BO_Assign) {
      const Expr *SelfRHS =
          lookThroughPtrToPtrCasts(BO->getRHS()->IgnoreParenImpCasts());
      if (const auto *RHSDRE = dyn_cast<DeclRefExpr>(SelfRHS))
        if (RHSDRE->getDecl() == VD)
          return;
    }
    State.NarrowedVars.erase(VD);
    State.NullableVars.erase(VD);
    forgetFactsAbout(VD);

    if (BO->getOpcode() == BO_Assign) {
      recordPointerSource(VD, BO->getRHS());
      recordTernaryPointerGuard(VD, BO->getRHS());
      recordPointerImplication(VD, BO->getRHS());
      storeToVar(VD, BO->getRHS(), BO);
    }
  }

  /// Store RHS into a pointer variable. &local additionally records the
  /// target so a later store through VD can invalidate it.
  void storeToVar(const VarDecl *VD, const Expr *RHS, const Expr *DiagExpr) {
    if (const auto *UO = dyn_cast<UnaryOperator>(RHS->IgnoreParenImpCasts()))
      if (UO->getOpcode() == UO_AddrOf) {
        narrowAsAddrOf(VD, UO);
        return;
      }
    storePointer(PtrRef{VD, std::nullopt}, RHS, DiagExpr);
  }

  /// Record Target's facts after it is assigned or initialized from RHS, and
  /// report a nullable value stored into a declared _Nonnull target. The
  /// caller has already dropped Target's old facts. Returns whether Target is
  /// now known non-null: from the value, or, when the value is unknown, from
  /// a _Nonnull declaration.
  bool storePointer(const PtrRef &Target, const Expr *RHS,
                    const Expr *DiagExpr) {
    bool DeclaredNonnull = isNonnullType(Target.getType());
    switch (classifyStoredValue(RHS)) {
    case StoredValue::Nullable:
      // A provably nullable value overrides the declared _Nonnull.
      State.markNullable(Target);
      if (DeclaredNonnull && Reporting && !isNullInitOfVar(Target, RHS)) {
        ++NumAssignmentWarnings;
        if (Target.VD)
          Handler.handleNullableAssignment(DiagExpr, Target.VD);
        else
          Handler.handleNullableMemberAssignment(DiagExpr,
                                                 Target.Path->leafField());
      }
      return false;
    case StoredValue::Nonnull:
      State.markNarrowed(Target);
      return true;
    case StoredValue::Unknown:
      if (!DeclaredNonnull)
        return false;
      State.markNarrowed(Target);
      return true;
    }
    llvm_unreachable("unhandled StoredValue");
  }

  /// Whether RHS is a null pointer constant initializing Target's variable.
  /// Sema reports that (warn_nullability_safety_null_init), including for
  /// globals the flow analysis never sees, so the flow report would be a
  /// duplicate.
  bool isNullInitOfVar(const PtrRef &Target, const Expr *RHS) const {
    return Target.VD && Target.VD->getInit() == RHS &&
           RHS->isNullPointerConstant(Ctx, Expr::NPC_ValueDependentIsNotNull);
  }

  /// Classify the value of any pointer store (variable init, assignment,
  /// member assignment, aggregate init). Flow facts about the
  /// value come first: its narrowing, then any taint, which beats a _Nonnull
  /// declaration (a _Nonnull variable assigned null reads as Nullable
  /// wherever it is copied). Only then do proofs from the expression's form
  /// or declared type count.
  StoredValue classifyStoredValue(const Expr *E) const {
    E = E->IgnoreParenImpCasts();
    if (isExprNarrowedNonnull(E))
      return StoredValue::Nonnull;
    if (isExprNullable(E))
      return StoredValue::Nullable;
    if (isNonnullInit(E))
      return StoredValue::Nonnull;
    return StoredValue::Unknown;
  }

  /// Whether VD is narrowed (proven non-null) in the current state.
  bool isNarrowed(const VarDecl *VD) const {
    return State.NarrowedVars.contains(VD);
  }

  /// Whether the member path is narrowed in the current state.
  bool isMemberNarrowed(const MemberAccessPath &Path) const {
    return State.NarrowedMembers.contains(Path);
  }

  /// Whether a smart pointer expression (the implicit object of operator->
  /// or operator*) is narrowed in the current state.
  bool isSmartPointerNarrowed(const Expr *E) const {
    auto R = PtrRef::fromExpr(E);
    return R && State.isNarrowed(*R);
  }

  /// Whether a smart pointer expression is flow-marked nullable.
  bool isSmartPointerNullable(const Expr *E) const {
    auto R = PtrRef::fromExpr(E);
    return R && State.isNullable(*R);
  }

  /// Warn on a smart pointer dereference unless the pointer is narrowed or
  /// declared _Nonnull. Flow facts after reset/move override the declared
  /// contract. Past the dereference the pointer is non-null on this path (a
  /// null one would not have got here), so only the first dereference on
  /// each path warns; a join keeps that only when every path dereferenced.
  /// Raw pointers keep warning at every dereference.
  void checkSmartPtrDeref(const Expr *DerefExpr, const Expr *Obj) {
    if (isSmartPointerMaybeNull(Obj))
      warnSmartPtrDeref(DerefExpr, Obj);
    if (auto R = smartPtrRef(Obj))
      State.markNarrowed(*R);
  }

  /// Whether the smart pointer Obj names is known to be possibly null right
  /// now: flow-marked nullable, or declared _Nullable and not narrowed.
  /// Unlike isSmartPointerMaybeNull this ignores the mode default, so it is
  /// what a copy or move carries to its destination.
  bool isSmartPointerKnownNullable(const Expr *Obj) const {
    if (isSmartPointerNullable(Obj))
      return true;
    return !isSmartPointerNarrowed(Obj) && isSmartPointerDeclaredNullable(Obj);
  }

  bool isSmartPointerMaybeNull(const Expr *Obj) const {
    if (isSmartPointerNullable(Obj))
      return true;
    if (isSmartPointerNarrowed(Obj))
      return false;
    if (isSmartPointerDeclaredNullable(Obj) ||
        getNullableSmartPtrCallResult(Obj))
      return true;
    if (isSmartPointerDeclaredNonnull(Obj))
      return false;
    return StrictNonnull ||
           Options.DefaultNullability != NullabilityKind::NonNull;
  }

  /// Whether Base (of a -> access) is an overloaded operator-> call. The
  /// dereference is checked when the receiver is a smart pointer; any other
  /// overloaded operator-> (iterators and the like) is not tracked.
  bool checkSmartPtrArrow(const Expr *DerefExpr, const Expr *Base) {
    const auto *OCE = dyn_cast<CXXOperatorCallExpr>(Base);
    if (!OCE || OCE->getOperator() != OO_Arrow)
      return false;
    if (OCE->getNumArgs() >= 1) {
      const Expr *Obj = OCE->getArg(0);
      if (isSmartPointerObject(Obj))
        checkSmartPtrDeref(DerefExpr, Obj);
    }
    return true;
  }

  /// Gate the built-in C library nullable-return list on the langopt so
  /// -fno-nullability-libc-nullable-returns fully disables it.
  bool isLibcNullableReturn(const CallExpr *CE) const {
    return Options.LibcNullableReturns && isLibcNullableReturnCall(CE);
  }

  /// Count and report a nullable dereference to the handler.
  void reportDeref(const Expr *DerefExpr, QualType PtrType) {
    if (Reporting) {
      ++NumDereferenceWarnings;
      Handler.handleNullableDereference(DerefExpr, PtrType);
    }
  }

  /// Report DerefExpr when PtrType (already adjusted by the caller for flow
  /// state and template sugar) is nullable.
  void checkDeref(const Expr *DerefExpr, QualType PtrType) {
    if (isNullableType(PtrType, Options.DefaultNullability)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  deref: nullable " << PtrType.getAsString() << "\n");
      reportDeref(DerefExpr, PtrType);
    }
  }

  /// Check dereference of a non-variable, non-member expression.
  /// Unwraps casts/arithmetic to avoid template-instantiation false
  /// positives where _Nullable is baked into cast result types.
  void checkExprDeref(const Expr *DerefExpr, const Expr *PtrExpr) {
    bool FoundCast = false;
    const Expr *Origin = unwrapCastsAndArithmetic(PtrExpr, FoundCast);

    // dynamic_cast<T*> yields null on a failed runtime check regardless of the
    // source's nullability, so it is judged as the origin even under further
    // casts or arithmetic. The narrowed idiom
    // if (auto *d = dynamic_cast<T*>(p)) derefs the VarDecl d, not the cast.
    if (const auto *DCE = dyn_cast<CXXDynamicCastExpr>(Origin)) {
      if (DCE->getType()->isPointerType()) {
        reportDeref(DerefExpr, DCE->getType());
        return;
      }
    }

    // *(p ? p : fallback): the merged type of a ternary inherits _Nullable
    // from an arm the condition guards, so judge the arms flow-sensitively.
    if (isa<AbstractConditionalOperator>(Origin)) {
      if (isExprNullable(Origin))
        reportDeref(DerefExpr, PtrExpr->getType());
      return;
    }
    // *p++, *(p + 1): the dereferenced value is p's, so use p's flow
    // state rather than the arithmetic expression's type.
    if (const auto *DRE = dyn_cast<DeclRefExpr>(Origin)) {
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        if (VD->getType()->isPointerType()) {
          if (!isNarrowed(VD))
            checkVarDeref(DerefExpr, VD);
          return;
        }
      }
    }

    if (isa<CXXThisExpr>(Origin))
      return;
    if (const auto *UO = dyn_cast<UnaryOperator>(Origin))
      if (UO->getOpcode() == UO_AddrOf)
        return;
    if (const auto *CE = dyn_cast<CallExpr>(Origin)) {
      // A return type drawn from a nullable template argument overrides the
      // STL and all-returns-nonnull skips.
      bool TemplateOverride = false;
      if (const auto *MCE = dyn_cast<CXXMemberCallExpr>(CE)) {
        QualType ArgTy = getTemplateArgTypeForMethodReturn(MCE);
        if (!ArgTy.isNull() && ArgTy->getNullability())
          TemplateOverride = true;
      }
      if (!TemplateOverride && isStlNonnullReturnCall(CE))
        return;
      if (const auto *Callee = CE->getDirectCallee()) {
        if (!TemplateOverride && isKnownAllReturnsNonnull(Callee))
          return;
      }
      // sp.get() on a narrowed smart pointer is non-null.
      if (const Expr *Obj = smartPtrGetReceiver(CE))
        if (isSmartPointerNarrowed(Obj))
          return;
    }
    // Throwing operator new never returns null.
    if (const auto *NE = dyn_cast<CXXNewExpr>(Origin)) {
      if (!NE->shouldNullCheckAllocation())
        return;
    }

    if (auto Path = decomposeMemberAccess(Origin)) {
      if (isMemberNarrowed(*Path))
        return;
    }

    QualType CheckTy = FoundCast ? Origin->getType() : PtrExpr->getType();
    // A type without nullability may still get it from a template argument on
    // a method return (Container<int*_Nullable>::get()).
    if (!CheckTy->getNullability()) {
      if (const auto *MCE = dyn_cast<CXXMemberCallExpr>(PtrExpr)) {
        QualType ArgTy = getTemplateArgTypeForMethodReturn(MCE);
        if (!ArgTy.isNull() && ArgTy->getNullability())
          CheckTy = ArgTy;
      }
    }
    checkDeref(DerefExpr, CheckTy);
  }

  /// Report a dereference of VD when its declared type or its flow state is
  /// nullable. Callers test isNarrowed first.
  void checkVarDeref(const Expr *DerefExpr, const VarDecl *VD) {
    QualType Ty = VD->getType();
    if (isNullableType(Ty, Options.DefaultNullability) ||
        State.NullableVars.contains(VD)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  deref: var '" << VD->getNameAsString() << "'\n");
      reportDeref(DerefExpr, Ty);
    }
  }

  /// Arithmetic on a pointer implies it is valid, so warn when the operand
  /// variable may be null and has not been narrowed.
  void checkVarArithmetic(const Expr *ArithExpr, const VarDecl *VD) {
    if (!isNarrowed(VD) &&
        (isNullableType(VD->getType(), Options.DefaultNullability) ||
         State.NullableVars.contains(VD))) {
      if (Reporting) {
        ++NumArithmeticWarnings;
        Handler.handleNullableArithmetic(ArithExpr, VD->getType(), VD);
      }
    }
  }

  /// Report a smart pointer dereference: a local always warns, a var.member
  /// path always warns, a this->member path warns only when flow marked it
  /// nullable (members set in constructors would otherwise warn), and a call
  /// result warns only when it may be null by contract.
  void warnSmartPtrDeref(const Expr *DerefExpr, const Expr *Obj) {
    auto R = PtrRef::fromExpr(Obj);
    if (!R) {
      if (const CallExpr *CE = getNullableSmartPtrCallResult(Obj)) {
        const FunctionDecl *Callee = CE->getDirectCallee();
        reportDeref(DerefExpr,
                    Callee ? Callee->getReturnType().getNonReferenceType()
                           : CE->getType());
      }
      return;
    }
    if (R->VD) {
      LLVM_DEBUG(llvm::dbgs() << "  deref: smart ptr '"
                              << R->VD->getNameAsString() << "'\n");
      reportDeref(DerefExpr, R->VD->getType());
      return;
    }
    // Callers have checked that Obj is a smart pointer object, so a path's
    // leaf may also be a reference member or a class deriving from one.
    if (R->Path->Root || State.NullableMembers.contains(*R->Path))
      reportDeref(DerefExpr, R->getType());
  }

  /// Remove any BoolGuards with a fact about the given pointer variable,
  /// directly or as the root of a member path.
  void invalidateBoolGuardsFor(const VarDecl *VD) {
    State.BoolGuards.remove_if([VD](const auto &Entry) {
      return llvm::any_of(Entry.second, [VD](const ConditionResult &CR) {
        return CR.Ref.VD == VD || (CR.Ref.Path && CR.Ref.Path->Root == VD);
      });
    });
  }

  /// Remove BoolGuards and member aliases that mention a member path under
  /// Prefix (the path was just assigned, so the facts are stale), and guards
  /// held in a field under Prefix.
  void invalidateGuardsAndAliasesWithPrefix(const MemberAccessPath &Prefix) {
    State.BoolGuards.remove_if([&Prefix](const auto &Entry) {
      if (Entry.first.Path && pathHasPrefix(*Entry.first.Path, Prefix))
        return true;
      return llvm::any_of(Entry.second, [&Prefix](const ConditionResult &CR) {
        return CR.Ref.Path && pathHasPrefix(*CR.Ref.Path, Prefix);
      });
    });
    State.MemberAliases.remove_if([&Prefix](const auto &Entry) {
      return pathHasPrefix(Entry.second, Prefix);
    });
  }

  /// True when the pointer-valued expression is known non-null right now:
  /// a narrowed or _Nonnull variable, or a narrowed member path, seen through
  /// pointer-to-pointer casts (never dynamic_cast).
  bool isExprNarrowedNonnull(const Expr *E) const {
    E = lookThroughPtrToPtrCasts(E->IgnoreParenImpCasts());
    auto R = PtrRef::fromExpr(E);
    if (!R)
      return false;
    if (const VarDecl *VD = R->VD)
      return isNarrowed(VD) ||
             (isNonnullType(VD->getType()) && !State.NullableVars.contains(VD));
    return isMemberNarrowed(*R->Path);
  }

  /// Record VD = <member path> so later narrowing of either side reaches
  /// the other. The path's current narrowed/nullable state is copied onto VD
  /// by the caller's isExprNarrowedNonnull/isExprNullable checks.
  void trackMemberCopy(const VarDecl *VD, const Expr *Init) {
    Init = lookThroughPtrToPtrCasts(Init->IgnoreParenImpCasts());
    auto Path = decomposeMemberAccess(Init);
    if (!Path || !Path->leafField()->getType()->isPointerType())
      return;
    // p = p->next: the path is rooted at the variable being overwritten,
    // so after the store it names a different object. Recording it would
    // let if (p) falsely narrow the new p's next.
    if (Path->Root == VD)
      return;
    State.MemberAliases[VD] = *Path;
  }

  /// Record where VD's new value came from after VD = RHS (or VD's
  /// initializer): an alias to another pointer variable, seen through
  /// explicit casts so T *y = static_cast<T *>(x) also aliases, and a copy
  /// of a member path.
  void recordPointerSource(const VarDecl *VD, const Expr *RHS) {
    RHS = RHS->IgnoreParenImpCasts();
    if (const auto *SrcDRE = dyn_cast<DeclRefExpr>(stripNonDynamicCasts(RHS)))
      if (const auto *SrcVD = dyn_cast<VarDecl>(SrcDRE->getDecl()))
        if (SrcVD->getType()->isPointerType())
          State.Aliases[VD] = resolveAlias(SrcVD);
    trackMemberCopy(VD, RHS);
  }

  /// VD = &local: VD is non-null, and remember the target so a later store
  /// through VD (*VD = x) can drop the target's narrowing.
  void narrowAsAddrOf(const VarDecl *VD, const UnaryOperator *AddrOf) {
    State.markNarrowed(VD);
    if (const auto *TgtDRE =
            dyn_cast<DeclRefExpr>(AddrOf->getSubExpr()->IgnoreParenImpCasts()))
      if (const auto *TgtVD = dyn_cast<VarDecl>(TgtDRE->getDecl()))
        State.AddrOfTargets[VD] = TgtVD;
  }

  /// Remove any Aliases that target the given pointer variable (the alias
  /// source was reassigned, so copies of its old value are stale).
  void invalidateAliasesFor(const VarDecl *VD) {
    State.Aliases.remove_if(
        [VD](const auto &Entry) { return Entry.second == VD; });
  }

  /// The alias target of VD (chains are flattened when recorded), or VD
  /// itself if it is not an alias of anything.
  const VarDecl *resolveAlias(const VarDecl *VD) const {
    auto It = State.Aliases.find(VD);
    return It != State.Aliases.end() ? It->second : VD;
  }

  /// Drop every narrowed/nullable member path and member alias rooted at VD.
  void invalidateMembersFor(const VarDecl *VD) {
    auto RootedAtVD = [VD](const MemberAccessPath &Path) {
      return Path.Root == VD;
    };
    State.NarrowedMembers.remove_if(RootedAtVD);
    State.NullableMembers.remove_if(RootedAtVD);
    State.MemberAliases.remove_if(
        [&](const auto &Entry) { return RootedAtVD(Entry.second); });
    State.BoolGuards.remove_if([&](const auto &Entry) {
      return Entry.first.Path && RootedAtVD(*Entry.first.Path);
    });
  }

  /// Drop everything known about VD, its own narrowed / nullable flags
  /// included, as when its declaration runs again.
  void forgetVariable(const VarDecl *VD) {
    State.clear(PtrRef{VD, std::nullopt});
    State.MustNullableVars.erase(VD);
    forgetFactsAbout(VD);
  }

  /// Drop the facts that named a smart pointer's old value when it is
  /// assigned, reset, released, swapped or moved from: the guards that
  /// tested it (bool ok = sp != nullptr), the member paths reached through it
  /// (sp->child), and, for a variable, everything forgetFactsAbout drops.
  /// Does not touch R's own narrowed/nullable flags.
  void forgetSmartPtrFacts(const PtrRef &R) {
    if (R.VD) {
      forgetFactsAbout(R.VD);
      return;
    }
    invalidateGuardsAndAliasesWithPrefix(*R.Path);
    // Paths reached through the old pointee (this->sp->child).
    auto Below = [&R](const MemberAccessPath &P) {
      return P.Fields.size() > R.Path->Fields.size() &&
             pathHasPrefix(P, *R.Path);
    };
    State.NarrowedMembers.remove_if(Below);
    State.NullableMembers.remove_if(Below);
  }

  /// Drop every fact that named VD's old value: member paths rooted at it,
  /// guards and aliases that mention it, and its own alias and address-of
  /// records. Does not touch VD's own narrowed/nullable flags.
  void forgetFactsAbout(const VarDecl *VD) {
    invalidateMembersFor(VD);
    invalidateBoolGuardsFor(VD);
    State.BoolGuards.erase(PtrRef::var(VD));
    invalidateAliasesFor(VD);
    State.Aliases.erase(VD);
    State.MemberAliases.erase(VD);
    State.AddrOfTargets.erase(VD);
  }

  /// Drop every narrowed/nullable member path under Prefix: assigning
  /// var.inner invalidates var.inner.x, var.inner.y, and so on.
  void invalidateMembersWithPrefix(const MemberAccessPath &Prefix) {
    auto HasPrefix = [&Prefix](const MemberAccessPath &Path) {
      return pathHasPrefix(Path, Prefix);
    };
    State.NarrowedMembers.remove_if(HasPrefix);
    State.NullableMembers.remove_if(HasPrefix);
    invalidateGuardsAndAliasesWithPrefix(Prefix);
  }

  /// In Cond ? T : F, an arm can be non-null only because the condition
  /// guards it: in p ? p : fallback the true arm is non-null even though p
  /// may be null. Returns whether Cond narrows the pointer named by Arm on the
  /// branch that selects it (TrueBranch selects the ? arm, otherwise the :
  /// arm).
  bool armNarrowedByCondition(const Expr *Arm, const Expr *Cond,
                              bool TrueBranch) const {
    if (!Arm || !Cond)
      return false;
    // GNU p ?: q hands the shared operand to the true arm as an
    // OpaqueValueExpr; look through it to the pointer it names.
    Arm =
        lookThroughPtrToPtrCasts(stripOpaqueValue(Arm->IgnoreParenImpCasts()));
    auto ArmRef = PtrRef::fromExpr(Arm);
    if (!ArmRef)
      return false;
    SmallVector<ConditionResult, 2> Results;
    analyzeCondition(Cond, Ctx, Results, &State.BoolGuards);
    for (const auto &CR : Results) {
      // A true-branch arm needs the pointer non-null when Cond is true
      // (Negated == false); a false-branch arm when Cond is false.
      if (CR.Negated == TrueBranch)
        continue;
      if (*ArmRef == CR.Ref)
        return true;
    }
    return false;
  }

  /// Whether an initializer is provably non-null: &x, new, this, a narrowed
  /// or _Nonnull variable, a cast of a non-null value, or arithmetic on one.
  bool isNonnullInit(const Expr *Init) const {
    if (!Init)
      return false;
    Init = stripOpaqueValue(Init->IgnoreParenImpCasts());
    // Ternary: both arms must be non-null, either unconditionally or because
    // the condition guards the arm (p ? p : fallback).
    if (const auto *CO = dyn_cast<AbstractConditionalOperator>(Init)) {
      const Expr *Cond = CO->getCond();
      bool TrueOK = isNonnullInit(CO->getTrueExpr()) ||
                    armNarrowedByCondition(CO->getTrueExpr(), Cond, true);
      bool FalseOK = isNonnullInit(CO->getFalseExpr()) ||
                     armNarrowedByCondition(CO->getFalseExpr(), Cond, false);
      return TrueOK && FalseOK;
    }
    if (const auto *DRE = dyn_cast<DeclRefExpr>(Init)) {
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
        if (isNonnullType(VD->getType()) || isNarrowed(VD))
          return true;
    }
    // Throwing operator new never returns null.
    if (const auto *NE = dyn_cast<CXXNewExpr>(Init)) {
      if (!NE->shouldNullCheckAllocation())
        return true;
    }
    // Pointer dynamic_cast may return null even for a non-null source.
    if (const auto *DCE = dyn_cast<CXXDynamicCastExpr>(Init))
      if (DCE->getType()->isPointerType())
        return false;
    // Other pointer-to-pointer casts preserve null/nonnull status.
    if (const auto *CE = dyn_cast<ExplicitCastExpr>(Init))
      return isNonnullInit(CE->getSubExpr());
    if (isa<CXXThisExpr>(Init))
      return true;
    // Pointer arithmetic on a non-null pointer is non-null.
    if (const auto *BO = dyn_cast<BinaryOperator>(Init)) {
      if (BO->getOpcode() == BO_Add || BO->getOpcode() == BO_Sub) {
        if (BO->getLHS()->getType()->isPointerType())
          return isNonnullInit(BO->getLHS()->IgnoreParenImpCasts());
        if (BO->getRHS()->getType()->isPointerType())
          return isNonnullInit(BO->getRHS()->IgnoreParenImpCasts());
      }
    }
    if (const auto *UO = dyn_cast<UnaryOperator>(Init)) {
      if (UO->getOpcode() == UO_AddrOf)
        return true;
    }
    // A call to a function whose every return is proven non-null, or to a
    // known non-null STL method. The C library nullable functions (malloc,
    // fopen, ...) never count.
    if (const auto *CE = dyn_cast<CallExpr>(Init)) {
      if (isLibcNullableReturn(CE))
        return false;
      if (isStlNonnullReturnCall(CE))
        return true;
      if (const auto *Callee = CE->getDirectCallee()) {
        if (isKnownAllReturnsNonnull(Callee))
          return true;
      }
      // sp.get() on a narrowed smart pointer returns nonnull. Falls through
      // when the receiver is neither a smart pointer variable nor a member
      // path.
      if (const Expr *Obj = smartPtrGetReceiver(CE))
        if (smartPtrRef(Obj))
          return isSmartPointerNarrowed(Obj);
    }
    // Anything else whose own type is _Nonnull (a call to T *_Nonnull f(),
    // a _Nonnull field). Checked last so the flow-sensitive cases above,
    // which can contradict the declared type, take precedence.
    return isNonnullType(Init->getType());
  }

  /// Warn when a nullable pointer reaches a nonnull parameter, then narrow
  /// the argument variable: surviving the call proves it non-null.
  void checkNonnullParamArg(const Expr *ArgExpr, const ParmVarDecl *Param) {
    const Expr *Arg = ArgExpr->IgnoreParenImpCasts();
    if (isExprNullable(Arg)) {
      if (Reporting) {
        ++NumArgumentWarnings;
        Handler.handleNullableArgument(ArgExpr, Param);
      }
    }
    if (const auto *DRE = dyn_cast<DeclRefExpr>(Arg)) {
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
        if (VD->getType()->isPointerType())
          State.markNarrowed(VD);
    }
  }

  /// Declared-type nullability. With ExplicitOnly only an explicit _Nullable
  /// counts; otherwise _Null_unspecified also counts under
  /// -fnullability-default=nullable.
  bool isNullableByType(QualType Ty, bool ExplicitOnly) const {
    if (ExplicitOnly)
      return isExplicitlyNullableType(Ty);
    if (StrictNonnull)
      return Ty->isPointerType() && !isNonnullType(Ty);
    return isNullableType(Ty, Options.DefaultNullability);
  }

  bool isProvablyNonnull(const Expr *E) const {
    llvm::SaveAndRestore Strict(StrictNonnull, true);
    return !isExprNullable(E);
  }

  NullabilityEvidence classifyEvidence(const Expr *E, bool IsNonnull) const {
    if (IsNonnull)
      return NullabilityEvidence::Nonnull;
    if (!isExprNullable(E, /*ExplicitOnly=*/true)) {
      if (const auto *CO =
              dyn_cast<AbstractConditionalOperator>(E->IgnoreParenImpCasts()))
        for (const Expr *Arm : {CO->getTrueExpr(), CO->getFalseExpr()})
          if (Arm->IgnoreParenCasts()->isNullPointerConstant(
                  Ctx, Expr::NPC_ValueDependentIsNotNull))
            return NullabilityEvidence::MaybeNull;
      return NullabilityEvidence::Unknown;
    }
    return isExprNullable(E, /*ExplicitOnly=*/true, /*MustOnly=*/true)
               ? NullabilityEvidence::Nullable
               : NullabilityEvidence::MaybeNull;
  }

  /// Whether E may be null, taking flow into account. Every site
  /// (initializer, assignment, argument, return, ternary dereference) uses
  /// this one judgment, so a tracked pointer's flow state wins over its
  /// declared type: narrowing overrides a declared _Nullable, and taint
  /// (assigned null, reset, moved-from) overrides a declared _Nonnull.
  ///
  /// With ExplicitOnly, used for evidence, only provably nullable sources
  /// count: an explicit _Nullable, a null constant, or flow-tracked nullable
  /// state (with MustOnly, only state nullable on every incoming path).
  /// Pointers merely defaulted to nullable do not count, so _Nullable is
  /// never inferred from them, and a ternary is judged by its merged type.
  bool isExprNullable(const Expr *E, bool ExplicitOnly = false,
                      bool MustOnly = false) const {
    if (!E)
      return false;
    E = E->IgnoreParenImpCasts();
    if (!ExplicitOnly)
      E = stripOpaqueValue(E);
    // Null pointer constants (nullptr, NULL, (T*)0) are always nullable.
    // The common type of a ternary like cond ? p : (T*)0 may strip the
    // qualifier, so the arm is judged directly rather than by the
    // expression's type.
    if (E->isNullPointerConstant(Ctx, Expr::NPC_ValueDependentIsNotNull))
      return true;
    ExplicitCastInfo Cast = getExplicitCastInfo(E);
    if (Cast.DefinitiveCast) {
      if (Cast.Operand->isNullPointerConstant(
              Ctx, Expr::NPC_ValueDependentIsNotNull))
        return true;
      if (Cast.isNonnull())
        return false;
      const Expr *Origin = stripNonDynamicCasts(E);
      if (const auto *DCE = dyn_cast<CXXDynamicCastExpr>(Origin))
        if (DCE->getType()->isPointerType())
          return true;
      if (Origin != E)
        return isExprNullable(Origin, ExplicitOnly, MustOnly);
    }
    // Pointer dynamic_cast is nullable even when its source is non-null, and
    // under the nonnull default its unannotated result type says nothing, so
    // this rule is what keeps T *q = dynamic_cast<T *>(p) from narrowing q.
    if (const auto *DCE = dyn_cast<CXXDynamicCastExpr>(E))
      if (DCE->getType()->isPointerType())
        return true;
    // Other casts preserve null-ness (static_cast<Base*>(this) is non-null).
    if (const auto *CE = dyn_cast<ExplicitCastExpr>(E))
      return isExprNullable(CE->getSubExpr(), ExplicitOnly, MustOnly);
    // Ternary: nullable iff some arm is nullable and the condition does not
    // guard that arm (p ? p : &x is non-null even though p is nullable,
    // because the true arm is only selected when p tested non-null). The
    // ExplicitOnly form judges a ternary by its merged type below.
    if (!ExplicitOnly) {
      if (const auto *CO = dyn_cast<AbstractConditionalOperator>(E)) {
        const Expr *Cond = CO->getCond();
        bool TrueNullable =
            isExprNullable(CO->getTrueExpr()) &&
            !armNarrowedByCondition(CO->getTrueExpr(), Cond, true);
        bool FalseNullable =
            isExprNullable(CO->getFalseExpr()) &&
            !armNarrowedByCondition(CO->getFalseExpr(), Cond, false);
        return TrueNullable || FalseNullable;
      }
    }
    // A tracked variable or member path: flow state first, then the declared
    // type of the variable or leaf field.
    if (auto R = PtrRef::fromExpr(E)) {
      if (State.isNarrowed(*R))
        return false;
      if (MustOnly ? State.isMustNullable(*R) : State.isNullable(*R))
        return true;
      return isNullableByType(R->getType(), ExplicitOnly);
    }
    // Throwing operator new never returns null; nothrow new can.
    if (const auto *NE = dyn_cast<CXXNewExpr>(E))
      return NE->shouldNullCheckAllocation();
    if (const auto *CE = dyn_cast<CallExpr>(E)) {
      // Stdlib nullable returns (malloc, fopen, etc.) are provably nullable.
      if (isLibcNullableReturn(CE))
        return true;
      if (isStlNonnullReturnCall(CE))
        return false;
      if (const auto *Callee = CE->getDirectCallee()) {
        if (isKnownAllReturnsNonnull(Callee))
          return false;
      }
      // sp.get() follows the smart pointer's flow state. Falls through when
      // the receiver is neither a smart pointer variable nor a member path.
      // An unnarrowed receiver is not provably nullable, so ExplicitOnly
      // falls through to the (unannotated) return type instead.
      if (const Expr *Obj = smartPtrGetReceiver(CE)) {
        if (smartPtrRef(Obj)) {
          if (isSmartPointerNarrowed(Obj))
            return false;
          if (!ExplicitOnly)
            return isSmartPointerMaybeNull(Obj);
        }
      }
    }
    if (const auto *UO = dyn_cast<UnaryOperator>(E)) {
      if (UO->getOpcode() == UO_AddrOf)
        return false;
    }
    if (isa<CXXThisExpr>(E))
      return false;
    // p + n and p - n carry p's null-ness (arithmetic on a nullable p is
    // diagnosed on its own), matching isNonnullInit. Without this the
    // unannotated arithmetic type decides, and container_of on a narrowed
    // pointer reads as nullable.
    if (const auto *BO = dyn_cast<BinaryOperator>(E))
      if (E->getType()->isPointerType() &&
          (BO->getOpcode() == BO_Add || BO->getOpcode() == BO_Sub))
        return isExprNullable(BO->getLHS()->getType()->isPointerType()
                                  ? BO->getLHS()
                                  : BO->getRHS(),
                              ExplicitOnly, MustOnly);
    // For non-variable expressions, fall back to the declared type.
    return isNullableByType(E->getType(), ExplicitOnly);
  }

  /// Dereference of a member path (p->q->r): a flow-marked nullable member
  /// overrides its declared _Nonnull, and template-argument sugar supplies
  /// nullability the instantiated field type lost.
  void checkMemberExprDeref(const Expr *DerefExpr, const MemberExpr *ME) {
    // For sp->raw the smart pointer was checked when sp->raw itself was
    // visited; here the dereferenced pointer is the raw field.
    if (auto Path = decomposeMemberAccess(ME)) {
      if (State.NullableMembers.contains(*Path)) {
        reportDeref(DerefExpr, ME->getType());
      } else if (!isMemberNarrowed(*Path)) {
        QualType CheckTy = ME->getType();
        if (!CheckTy->getNullability()) {
          QualType ArgTy = getTemplateArgTypeForField(ME);
          if (!ArgTy.isNull() && ArgTy->getNullability())
            CheckTy = ArgTy;
        }
        checkDeref(DerefExpr, CheckTy);
      }
    }
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Seed the analysis's entry state: narrow parameters that are provably
/// non-null on entry (declared _Nonnull, __attribute__((nonnull)), or a
/// lambda's pointer params, which default to nonnull).
static NullState seedEntryState(const Decl *D) {
  NullState InitState;
  const auto *FD = dyn_cast_or_null<FunctionDecl>(D);
  if (!FD)
    return InitState;

  // Collect parameters declared nonnull via __attribute__((nonnull)),
  // either the whole-function form (applies to every pointer param) or
  // the indexed form (nonnull(N...), 1-based).
  llvm::SmallPtrSet<const ParmVarDecl *, 4> AttrNonnull;
  for (const auto *NNA : FD->specific_attrs<NonNullAttr>()) {
    if (NNA->args_size() == 0) {
      for (const auto *P : FD->parameters())
        if (P->getType()->isPointerType())
          AttrNonnull.insert(P);
    } else {
      for (const ParamIdx &Idx : NNA->args()) {
        unsigned I = Idx.getASTIndex();
        if (I < FD->getNumParams())
          AttrNonnull.insert(FD->getParamDecl(I));
      }
    }
  }
  // Lambda pointer parameters default to non-null unless explicitly
  // _Nullable. A lambda's callers control what it receives, so a null
  // argument is reported at the call site instead (see checkCallArguments).
  bool IsLambda = false;
  if (const auto *MD = dyn_cast<CXXMethodDecl>(FD))
    IsLambda = MD->getParent()->isLambda();

  for (const auto *Param : FD->parameters()) {
    if (!Param->getType()->isPointerType())
      continue;
    if (isNonnullType(Param->getType()) || AttrNonnull.contains(Param) ||
        (IsLambda && !isExplicitlyNullableType(Param->getType())))
      InitState.markNarrowed(Param);
  }
  return InitState;
}

/// Narrow every operand of a whole && / || branch condition: on the true edge
/// of A && B all operands held, and on the false edge of A || B all failed.
/// The CFG routes each operand's edge straight to the branch only when the
/// && / || is the condition itself. Behind a bool cast (the C++ assert macro
/// is static_cast<bool>(p && "why") ? ... : __assert_fail(...)) or a cleanup,
/// the operands first merge into one value, and the per-operand narrowing is
/// lost unless it is recovered here.
static void narrowWholeCondition(const Expr *Cond, const NullState &State,
                                 ASTContext &Ctx, NullState &TrueState,
                                 NullState &FalseState) {
  if (!Cond)
    return;
  const Expr *E = Cond;
  while (true) {
    const Expr *Next = ignoreExplicitBoolCast(E->IgnoreParenImpCasts());
    if (const auto *EWC = dyn_cast<ExprWithCleanups>(Next))
      Next = EWC->getSubExpr();
    if (Next == E)
      break;
    E = Next;
  }
  const auto *BO = dyn_cast<BinaryOperator>(E);
  if (!BO)
    return;
  if (BO->getOpcode() == BO_LAnd) {
    SmallVector<ConditionResult, 2> AndResults;
    decomposeChain(BO, BO_LAnd, Ctx, AndResults, &State.BoolGuards);
    for (const auto &CR : AndResults)
      if (!CR.Negated)
        applyNarrowing(TrueState, CR);
  } else if (BO->getOpcode() == BO_LOr) {
    SmallVector<ConditionResult, 2> OrResults;
    decomposeChain(BO, BO_LOr, Ctx, OrResults, &State.BoolGuards);
    for (const auto &CR : OrResults)
      if (CR.Negated)
        applyNarrowing(FalseState, CR);
  }
}

/// Apply branch-condition narrowing to a block's outgoing edges. TrueState
/// and FalseState start as copies of the block's exit state; each fact from
/// the terminator's condition (an if, loop, &&/||, or ?:) narrows its pointer
/// on the edge where the fact holds.
static void narrowOnTerminator(const CFGBlock *Block, const NullState &State,
                               ASTContext &Ctx, NullState &TrueState,
                               NullState &FalseState) {
  const Stmt *Term = Block->getTerminatorStmt();
  if (!Term)
    return;

  const Expr *WholeCond = nullptr;
  if (const auto *IS = dyn_cast<IfStmt>(Term))
    WholeCond = IS->getCond();
  else if (const auto *WS = dyn_cast<WhileStmt>(Term))
    WholeCond = WS->getCond();
  else if (const auto *FS = dyn_cast<ForStmt>(Term))
    WholeCond = FS->getCond();
  else if (const auto *DS = dyn_cast<DoStmt>(Term))
    WholeCond = DS->getCond();
  else if (const auto *CO = dyn_cast<AbstractConditionalOperator>(Term))
    WholeCond = CO->getCond(); // Covers GNU p ?: q as well.
  narrowWholeCondition(WholeCond, State, Ctx, TrueState, FalseState);

  const Expr *Cond = nullptr;
  if (WholeCond)
    Cond = getTerminalCondition(WholeCond);
  else if (const auto *BO = dyn_cast<BinaryOperator>(Term))
    if (BO->getOpcode() == BO_LAnd || BO->getOpcode() == BO_LOr)
      Cond = getTerminalCondition(BO->getLHS());

  if (Cond) {
    SmallVector<ConditionResult, 2> Results;
    analyzeCondition(Cond, Ctx, Results, &State.BoolGuards);
    for (const auto &CR : Results)
      applyNarrowing(CR.Negated ? FalseState : TrueState, CR);
  }
}

/// After the fixpoint, emit the all-returns-nonnull summary. Skipped if the
/// visit cap fired (a partial run cannot be trusted) and when the return type
/// is explicitly _Nullable: the annotation wins over body inference, since the
/// body may see _Nonnull members that are null at runtime.
static void emitAllReturnsNonnullSummary(const Decl *D, bool HitVisitCap,
                                         const ReturnSummary &Returns,
                                         NullabilitySafetyHandler &Handler) {
  if (HitVisitCap)
    return;
  const auto *FD = dyn_cast_or_null<FunctionDecl>(D);
  if (!FD || !FD->getReturnType()->isPointerType() ||
      !FD->getDeclName().isIdentifier())
    return;
  if (Returns.HasPointerReturn && Returns.AllNonnull &&
      !isExplicitlyNullableType(FD->getReturnType()))
    Handler.handleAllReturnsNonnull(FD);
}

/// Whether a declared parameter or return type carries a nullability
/// annotation. For a reference the annotation sits on the referenced type:
/// const std::shared_ptr<T> _Nonnull &.
static bool declaresNullability(QualType T) {
  return !T.isNull() && !T->isDependentType() &&
         T.getNonReferenceType()->getNullability();
}

static bool functionHasNullabilityAnnotations(const FunctionDecl *FD) {
  if (!FD || FD->isInvalidDecl())
    return false;

  // Annotations may be on another declaration (such as a header prototype)
  // while the definition is unannotated, so check the whole redeclaration
  // chain; otherwise a function that opted in through its prototype would be
  // skipped.
  for (const FunctionDecl *Redecl : FD->redecls()) {
    if (declaresNullability(Redecl->getReturnType()))
      return true;

    // During early function processing parameters might not be set up yet,
    // so guard with param_empty().
    if (!Redecl->param_empty()) {
      for (const ParmVarDecl *Param : Redecl->parameters())
        if (Param && declaresNullability(Param->getType()))
          return true;
    }
  }

  return false;
}

bool clang::hasExplicitNullabilityAnnotations(const Decl *D) {
  if (const auto *FD = dyn_cast_or_null<FunctionDecl>(D))
    return functionHasNullabilityAnnotations(FD);

  if (const auto *MD = dyn_cast_or_null<ObjCMethodDecl>(D)) {
    if (declaresNullability(MD->getReturnType()))
      return true;
    for (const ParmVarDecl *P : MD->parameters())
      if (P && declaresNullability(P->getType()))
        return true;
    return false;
  }
  if (const auto *BD = dyn_cast_or_null<BlockDecl>(D)) {
    for (const ParmVarDecl *P : BD->parameters())
      if (P && declaresNullability(P->getType()))
        return true;
    if (const TypeSourceInfo *TSI = BD->getSignatureAsWritten())
      if (const auto *FPT = TSI->getType()->getAs<FunctionProtoType>())
        if (declaresNullability(FPT->getReturnType()))
          return true;
    return false;
  }
  return false;
}

const Decl *clang::getNullabilitySafetyDefinition(const Decl *D) {
  const Decl *Def = nullptr;
  if (const auto *FD = dyn_cast_or_null<FunctionDecl>(D))
    Def = FD->getDefinition();
  else if (const auto *MD = dyn_cast_or_null<ObjCMethodDecl>(D))
    Def = MD->hasBody() ? MD : nullptr;
  else if (const auto *BD = dyn_cast_or_null<BlockDecl>(D))
    Def = BD;
  if (!Def || Def->isInvalidDecl() ||
      Def->getDeclContext()->isDependentContext())
    return nullptr;
  return Def;
}

namespace {
class TUDriverHandler final : public NullabilitySafetyHandler,
                              public NullabilitySafetySummaries {
  NullabilitySafetyHandler &Inner;
  llvm::DenseSet<const FunctionDecl *> AllReturnsNonnull;

public:
  bool InRecursiveSCC = false;

  explicit TUDriverHandler(NullabilitySafetyHandler &Inner) : Inner(Inner) {}

  void handleNullableDereference(const Expr *DerefExpr,
                                 QualType PtrType) override {
    Inner.handleNullableDereference(DerefExpr, PtrType);
  }
  void handleNullableArithmetic(const Expr *ArithExpr, QualType PtrType,
                                const VarDecl *VD) override {
    Inner.handleNullableArithmetic(ArithExpr, PtrType, VD);
  }
  void handleNullableReturn(const Expr *ReturnExpr) override {
    Inner.handleNullableReturn(ReturnExpr);
  }
  void handleNullableAssignment(const Expr *AssignExpr,
                                const VarDecl *LHSVar) override {
    Inner.handleNullableAssignment(AssignExpr, LHSVar);
  }
  void handleNullableMemberAssignment(const Expr *AssignExpr,
                                      const FieldDecl *Member) override {
    Inner.handleNullableMemberAssignment(AssignExpr, Member);
  }
  void handleNullableArgument(const Expr *ArgExpr,
                              const ParmVarDecl *Param) override {
    Inner.handleNullableArgument(ArgExpr, Param);
  }
  void handleNullableMemberAtExit(SourceLocation Loc,
                                  const FieldDecl *Member) override {
    Inner.handleNullableMemberAtExit(Loc, Member);
  }
  void handleMemberAssignEvidence(const Expr *AssignExpr,
                                  const FieldDecl *Member,
                                  NullabilityEvidence Kind) override {
    Inner.handleMemberAssignEvidence(AssignExpr, Member, Kind);
  }
  void handleReturnEvidence(const Expr *RetExpr, const FunctionDecl *Func,
                            NullabilityEvidence Kind) override {
    Inner.handleReturnEvidence(RetExpr, Func, Kind);
  }
  void handleParameterEvidence(const Expr *ArgExpr, const ParmVarDecl *Param,
                               const FunctionDecl *Func,
                               NullabilityEvidence Kind) override {
    Inner.handleParameterEvidence(ArgExpr, Param, Func, Kind);
  }
  void handleAllReturnsNonnull(const FunctionDecl *Func) override {
    if (InRecursiveSCC)
      return;
    AllReturnsNonnull.insert(Func->getCanonicalDecl());
    Inner.handleAllReturnsNonnull(Func);
  }
  void startFunction(const Decl *Def) override { Inner.startFunction(Def); }
  void finishFunction(const Decl *Def) override { Inner.finishFunction(Def); }

  bool isKnownAllReturnsNonnull(const FunctionDecl *Func) const override {
    return Func && AllReturnsNonnull.contains(Func->getCanonicalDecl());
  }
};
} // namespace

NullabilitySafetyOptions
NullabilitySafetyOptions::fromLangOptions(const LangOptions &LO) {
  NullabilitySafetyOptions Options;
  Options.DefaultNullability = LO.getNullabilityDefault();
  Options.LibcNullableReturns = LO.NullabilityLibcNullableReturns;
  return Options;
}

bool clang::isNullabilitySafetyOptedIn(
    const Decl *D, const NullabilitySafetyOptions &Options) {
  return Options.DefaultNullability != NullabilityKind::Unspecified ||
         hasExplicitNullabilityAnnotations(D);
}

void clang::runNullabilitySafetyOnTU(
    TranslationUnitDecl *TU, NullabilitySafetyHandler &Handler,
    const NullabilitySafetyOptions &Options,
    llvm::function_ref<bool(const Decl *)> Filter) {
  CallGraph CG;
  CG.addToCallGraph(TU);
  TUDriverHandler Driver(Handler);
  for (auto SCCI = llvm::scc_begin(&CG); !SCCI.isAtEnd(); ++SCCI) {
    Driver.InRecursiveSCC = SCCI.hasCycle();
    for (CallGraphNode *Node : *SCCI) {
      const Decl *Def = getNullabilitySafetyDefinition(Node->getDecl());
      if (!Def || !isNullabilitySafetyOptedIn(Def, Options) || !Filter(Def))
        continue;
      AnalysisDeclContext AC(nullptr, Def);
      AC.getCFGBuildOptions().setAllAlwaysAdd();
      if (!AC.getCFG())
        continue;
      Driver.startFunction(Def);
      runNullabilitySafetyAnalysis(AC, Driver, Options, &Driver);
      Driver.finishFunction(Def);
    }
  }
}

static void reportNonnullMembersNullAtExit(
    const Decl *D, const CFG &Cfg,
    const llvm::DenseMap<unsigned, NullState> &BlockEntryStates,
    NullabilitySafetyHandler &Handler) {
  const auto *MD = dyn_cast_or_null<CXXMethodDecl>(D);
  if (!MD || MD->isStatic() || isa<CXXDestructorDecl>(MD) ||
      MD->getParent()->isLambda() || MD->getRefQualifier() == RQ_RValue ||
      !MD->hasBody())
    return;
  auto It = BlockEntryStates.find(Cfg.getExit().getBlockID());
  if (It == BlockEntryStates.end())
    return;
  const NullState &Exit = It->second;
  SmallVector<const FieldDecl *, 4> Members;
  for (const MemberAccessPath &Path : Exit.MustNullableMembers) {
    if (Path.Root || Path.Fields.size() != 1)
      continue;
    const FieldDecl *FD = Path.leafField();
    if (FD->getParent() == MD->getParent() &&
        isSmartPointerType(FD->getType()) && isNonnullType(FD->getType()) &&
        Exit.isMustNullable(PtrRef{nullptr, Path}))
      Members.push_back(FD);
  }
  llvm::sort(Members, [](const FieldDecl *A, const FieldDecl *B) {
    return A->getFieldIndex() < B->getFieldIndex();
  });
  for (const FieldDecl *FD : Members)
    Handler.handleNullableMemberAtExit(MD->getBody()->getEndLoc(), FD);
}

void clang::runNullabilitySafetyAnalysis(
    AnalysisDeclContext &AC, NullabilitySafetyHandler &Handler,
    const NullabilitySafetyOptions &Options,
    const NullabilitySafetySummaries *Summaries) {
  CFG *Cfg = AC.getCFG();
  if (!Cfg)
    return;

  ReturnSummary Returns;

  ++NumFunctionsAnalyzed;
  ASTContext &Ctx = AC.getASTContext();
  // Cached per function; see isStdMoveInsideSmartPtrTransferCtx.
  const ParentMap &PM = AC.getParentMap();
  LLVM_DEBUG({
    if (const auto *ND = dyn_cast_or_null<NamedDecl>(AC.getDecl()))
      llvm::dbgs() << "nullability-safety: analyzing '" << ND->getNameAsString()
                   << "' (" << Cfg->size() << " blocks)\n";
  });

  using EdgeKey = std::pair<unsigned, unsigned>;
  llvm::DenseMap<EdgeKey, NullState> EdgeStates;
  llvm::DenseMap<unsigned, NullState> BlockEntryStates;

  ForwardDataflowWorklist Worklist(*Cfg, AC);

  const CFGBlock &Entry = Cfg->getEntry();
  const auto *EnclosingFunc = dyn_cast_or_null<FunctionDecl>(AC.getDecl());
  NullState InitState = seedEntryState(AC.getDecl());

  BlockEntryStates[Entry.getBlockID()] = InitState;
  Worklist.enqueueBlock(&Entry);

  // Termination safety net for the non-monotone lattice (see file overview).
  // Generous: well-behaved functions converge in a few visits per block.
  const unsigned MaxBlockVisits = Cfg->size() * 64;
  unsigned BlockVisits = 0;
  bool HitVisitCap = false;
  while (const CFGBlock *Block = Worklist.dequeue()) {
    if (++BlockVisits > MaxBlockVisits) {
      ++NumFixpointBailouts;
      HitVisitCap = true;
      break;
    }
    unsigned BlockID = Block->getBlockID();
    ++NumBlocksProcessed;
    LLVM_DEBUG(llvm::dbgs() << "  block B" << BlockID << " (preds:");

    NullState State;
    bool FirstPred = true;

    if (BlockID == Entry.getBlockID()) {
      State = BlockEntryStates[BlockID];
      FirstPred = false;
    }

    for (const CFGBlock *Pred : Block->preds()) {
      if (Pred) {
        LLVM_DEBUG(llvm::dbgs() << " B" << Pred->getBlockID());
        EdgeKey EK = {Pred->getBlockID(), BlockID};
        auto It = EdgeStates.find(EK);
        if (It != EdgeStates.end()) {
          if (FirstPred) {
            State = It->second;
            FirstPred = false;
          } else {
            State = join(State, It->second);
          }
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << ")\n");

    if (FirstPred)
      continue;

    // Skip a block whose entry state is unchanged: its outgoing edges would
    // not change either. The entry block is exempt because its pre-seeded
    // state would always match.
    if (BlockID != Entry.getBlockID()) {
      auto OldIt = BlockEntryStates.find(BlockID);
      if (OldIt != BlockEntryStates.end() && OldIt->second == State) {
        LLVM_DEBUG(llvm::dbgs() << "    converged, skipping\n");
        continue;
      }
    }
    BlockEntryStates[BlockID] = State;

    TransferFunctions TF(State, Handler, Returns, Ctx, Options, Summaries,
                         EnclosingFunc, &PM,
                         /*Reporting=*/false);
    for (const auto &Elem : *Block) {
      if (std::optional<CFGStmt> CS = Elem.getAs<CFGStmt>())
        if (const Stmt *S = CS->getStmt())
          TF.Visit(S);
    }

    NullState TrueState = State;
    NullState FalseState = State;
    narrowOnTerminator(Block, State, Ctx, TrueState, FalseState);

    for (const auto &[SucIdx, Succ] : llvm::enumerate(Block->succs())) {
      if (const CFGBlock *SuccBlock = Succ) {
        const NullState &SuccState =
            (Block->succ_size() == 2) ? (SucIdx == 0 ? TrueState : FalseState)
                                      : State;
        EdgeKey EK = {BlockID, SuccBlock->getBlockID()};
        auto It = EdgeStates.find(EK);
        if (It == EdgeStates.end() || It->second != SuccState) {
          LLVM_DEBUG(llvm::dbgs()
                     << "    edge B" << BlockID << "->B"
                     << SuccBlock->getBlockID() << " changed, enqueuing\n");
          EdgeStates[EK] = SuccState;
          Worklist.enqueueBlock(SuccBlock);
        }
      }
    }
  }

  if (const auto *CD = dyn_cast_or_null<CXXConstructorDecl>(AC.getDecl())) {
    NullState State = InitState;
    TransferFunctions(State, Handler, Returns, Ctx, Options, Summaries,
                      EnclosingFunc, &PM,
                      /*Reporting=*/true)
        .reportCtorInitEvidence(CD);
  }

  // Reporting pass: each reached block once, from its final entry state. An
  // intermediate state can be more optimistic than the converged one (a
  // back-edge taint has not arrived yet), so reporting during the fixpoint
  // would emit facts the final states contradict. After a visit-cap bailout
  // the states are the last ones observed.
  for (const CFGBlock *Block : *Cfg) {
    auto It = BlockEntryStates.find(Block->getBlockID());
    if (It == BlockEntryStates.end())
      continue;
    NullState State = It->second;
    TransferFunctions TF(State, Handler, Returns, Ctx, Options, Summaries,
                         EnclosingFunc, &PM,
                         /*Reporting=*/true);
    for (const auto &Elem : *Block)
      if (std::optional<CFGStmt> CS = Elem.getAs<CFGStmt>())
        if (const Stmt *S = CS->getStmt())
          TF.Visit(S);
  }

  reportNonnullMembersNullAtExit(AC.getDecl(), *Cfg, BlockEntryStates, Handler);
  emitAllReturnsNonnullSummary(AC.getDecl(), HitVisitCap, Returns, Handler);
}
