//===--- DynamicCountPointerAssignmentAnalysis.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for dynamic count pointer assignments
//  for -fbounds-safety based on CFGBlock analysis.
//
//===----------------------------------------------------------------------===//

#include "DynamicCountPointerAssignmentAnalysis.h"
#include "clang/AST/Attr.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/IgnoreExpr.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>

using namespace clang;

namespace {
/// This is to generate a unique key for members in nested structs. This is
/// necessary to distinguish member accesses of nested structs in different
/// parent structs as well as members of same structs from different instances.
/// @code
/// struct nested { int i; };
/// struct parent { struct nested n1; struct nested n2; };
/// struct parent p1;
/// struct parent p2;
/// @endcode
/// For the above example, this function distinguishes member accesses "p1.n1.i"
/// and "p1.n2.i", and "p1.n1.i" and "p2.n1.i".
void computeMemberExprKey(const Expr *E, llvm::raw_string_ostream &OS,
                          bool Top = true) {
  E = E->IgnoreParenCasts();
  if (auto ME = dyn_cast<MemberExpr>(E)) {
    computeMemberExprKey(ME->getBase(), OS, false);
    OS << (ME->isArrow() ? "->" : ".");
    if (!Top) {
      OS << ME->getMemberDecl()->getName();
    }
    return;
  } else if (auto DE = dyn_cast<DeclRefExpr>(E)) {
    OS << DE->getDecl()->getDeclName();
  } else if (auto OVE = dyn_cast<OpaqueValueExpr>(E)) {
    return computeMemberExprKey(OVE->getSourceExpr(), OS, Top);
  }
}

/// AssignedDeclRefResult - This represents result of analysis for a declaration
/// reference in LHS or RHS of variable assignment operation in terms of dynamic
/// counts or dynamic count pointers including params, locals, and fields.
/// @code
/// struct S { int *__counted_by(n) ptr; int n; };
/// S s;
/// s.ptr = ptrVal;
/// s.n = intVal;
/// @endcode
/// For instance, this holds information whether "s.ptr" in the assignment
/// operation is dynamic count or dynamic count pointer and ditto for "s.n".
struct AssignedDeclRefResult {
  /// The decl reference should be tracked. It is dynamic count or dynamic count
  /// pointer.
  unsigned IsTrackedVar : 1;
  /// The decl reference is dynamic range pointer.
  unsigned IsRangePtrVar : 1;
  /// The decl reference is dynamic count.
  unsigned IsCountVar : 1;
  /// The decl reference is dynamic count pointer.
  unsigned IsCountPtrVar : 1;
  /// The dynamic count is out parameter.
  unsigned IsOutCount : 1;
  /// The dynamic count pointer is out parameter.
  unsigned IsOutBuf : 1;
  /// The decl is a single pointer to struct with flexible array member.
  unsigned IsFlexBase : 1;
  /// The decl is a nested struct containing a dynamic count decl
  unsigned IsInnerStruct : 1;
  unsigned IsCountForFam : 1;
  /// The decl is a dependent param that is referred to by the return type that
  /// is a bounds-attributed type.
  unsigned IsDependentParamOfReturnType : 1;
  /// Dereference level of decl reference. Dereference level of decl reference.
  /// E.g., "**p = " has Level 2 while "p = " has zero.
  unsigned Level : 32 - 10;

  /// The decl actually referenced in the analyzed expression.
  ValueDecl *ThisVD = nullptr;
  /// Partial key calculation of DeclGroup in which ThisVD belongs.
  std::string Key;
  /// Pointer to struct with fam as a key.
  std::string FlexBaseKey;
  /// Type of assignment expression.
  QualType Ty;
  /// Decl of struct containing flexible array member.
  RecordDecl *FlexBaseDecl = nullptr;
  /// Loc of decl reference expression.
  SourceLocation Loc;

  AssignedDeclRefResult() {
    IsTrackedVar = 0;
    IsRangePtrVar = 0;
    IsCountPtrVar = 0;
    IsCountVar = 0;
    IsOutCount = 0;
    IsOutBuf = 0;
    IsFlexBase = 0;
    IsInnerStruct = 0;
    IsCountForFam = 0;
    IsDependentParamOfReturnType = 0;
    Level = 0;
  }
};

static ExprResult CastToCharPointer(Sema &S, Expr *PointerToCast) {
  SourceLocation Loc = PointerToCast->getBeginLoc();
  auto PtrTy = PointerToCast->getType()->getAs<PointerType>();
  auto SQT = PtrTy->getPointeeType().split();
  SQT.Ty = S.Context.CharTy.getTypePtr();
  QualType QualifiedChar = S.Context.getQualifiedType(SQT);
  QualType CharPtrTy = S.Context.getPointerType(
      QualifiedChar, PtrTy->getPointerAttributes());
  return S.BuildCStyleCastExpr(
      Loc, S.Context.getTrivialTypeSourceInfo(CharPtrTy), Loc, PointerToCast);
}

/// getInnerType - This returns "Level" nested pointee type.
QualType getInnerType(QualType Ty, unsigned Level) {
  for (unsigned i = 0; i < Level; ++i) {
    if (!Ty->isPointerType())
      return QualType();
    Ty = Ty->getPointeeType();
  }
  return Ty;
}

bool isValueDeclOutBuf(const ValueDecl *VD) {
  QualType Ty = VD->getType();
  return isa<ParmVarDecl>(VD) && Ty->isPointerType() &&
         Ty->getPointeeType()->isBoundsAttributedType();
}

bool isValueDeclOutCount(const ValueDecl *VD) {
  const auto *Att = VD->getAttr<DependerDeclsAttr>();
  return isa<ParmVarDecl>(VD) && Att && Att->getIsDeref();
}

bool isSinglePtrToStructWithFam(Sema &S, QualType Ty) {
  if (!Ty->isSinglePointerType() || Ty->isBoundsAttributedType())
    return false;

  FlexibleArrayMemberUtils FlexUtils(S);
  return FlexUtils.GetFlexibleRecord(Ty->getPointeeType());
}

/// analyzeAssignedDeclCommon - This shares analysis logic for declaration
/// reference from assignment expression and variable declaration statement.
void analyzeAssignedDeclCommon(Sema &S, ValueDecl *VD, QualType Ty,
                               AssignedDeclRefResult &Result) {
  if (Ty->isCountAttributedType()) {
    Result.ThisVD = VD;
    Result.IsTrackedVar = 1;
    Result.IsCountPtrVar = 1;
    return;
  }

  if (Result.Level == 0 && isa<ParmVarDecl>(VD) && Ty->isPointerType() &&
             Ty->getPointeeType()->isCountAttributedType()) {
    Result.IsOutBuf = 1;
    return;
  }

  const auto *Attr = VD->getAttr<DependerDeclsAttr>();

  const BoundsAttributedType *RetType = nullptr;
  const TypeCoupledDeclRefInfo *Info = nullptr;
  bool IsDepParamOfRetType = VD->isDependentParamOfReturnType(&RetType, &Info);
  if (IsDepParamOfRetType) {
    Result.IsDependentParamOfReturnType = 1;
    Result.ThisVD = VD;
  }

  bool IsDependentCount =
      Attr || (IsDepParamOfRetType && isa<CountAttributedType>(RetType));
  if (IsDependentCount) {
    bool IsDeref = Attr ? Attr->getIsDeref() : Info->isDeref();
    if (Result.Level == 0 && isa<ParmVarDecl>(VD) && IsDeref) {
      Result.IsOutCount = 1;
    } else if (Result.Level < 2) {
      // If the VD doesn't have a DependerDeclsAttr, this VD is used only in the
      // return type. For example:
      //   int *__counted_by(count) foo(int count);
      // In that case, the VD can be freely updated and we won't generate a
      // bounds-check for the updates, so set IsTrackedVar to false.
      Result.IsTrackedVar = !!Attr;
      Result.ThisVD = VD;
      if (Ty->isIntegralOrEnumerationType())
        Result.IsCountVar = 1;
      else
        Result.IsInnerStruct = 1;
    } else
      llvm_unreachable(
          "Dependent count reference can only be a single-level indirection");
    return;
  }

  if (Ty->isDynamicRangePointerType()) {
    Result.ThisVD = VD;
    Result.IsTrackedVar = 1;
    Result.IsRangePtrVar = 1;
    return;
  }

  if (isSinglePtrToStructWithFam(S, Ty)) {
    Result.ThisVD = VD;
    Result.IsTrackedVar = 1;
    Result.IsFlexBase = 1;
    return;
  }
}

/// analyzeVarDecl - This function analyzes variable declaration like
/// "int *__counted_by(len) ptr;" or "int len;".
void analyzeVarDecl(Sema &S, VarDecl *Var, AssignedDeclRefResult &Result) {
  Result.Ty = Var->getType();
  Result.Loc = Var->getLocation();
  analyzeAssignedDeclCommon(S, Var, Result.Ty, Result);
}

/// Check whether the dep group contains a flexible array member,
/// and update FlexBaseKey and FlexBaseDecl if so, for matching
/// with the relevant DepGroup.
/// Because counted_by pointers can share count fields with the
/// FAM, this relation can be indirect.
/// In the code below, assigning to \c SharedCount::p requires assigning
/// to \c SharedCount::len, which in turns requires assigning to the
/// flexbase pointer from a wide pointer:
/// \code
/// struct SharedCount {
///     int * __counted_by(len) p;
///     int len;
///     int fam[__counted_by(len)];
/// };
/// \endcode
///
/// Fields in structs that are nested in other structs can be referred
/// to from fields in the outer structs. Care must be taken to only check
/// the dependent fields when the inner struct type is actually used in the
/// context of the outer struct. This is checked with \c isParentStructOf().
/// Example of irrelevant dep decl ( \c SimpleOuter1::fam):
/// \code
/// struct SimpleInner {
///     int dummy;
///     int len;
/// };
/// struct SimpleOuter1 {
///     struct SimpleInner hdr;
///     char fam[__counted_by(hdr.len)];
/// };
/// void set_len(struct SimpleInner * p) {
///     // struct SimpleInner doesn't contain any FAMs, so not checked
///     // -fbounds-safety doesn't allow taking address of
///     // \c SimpleOuter1::hdr though, so this is never referred to by
///     // a FAM and thus safe
///     p->len = 2;
/// }
/// struct SimpleOuter2 {
///     struct SimpleInner hdr;
///     char fam[__counted_by(hdr.len)];
/// };
/// struct SimpleOuter2 *foo(int len) {
///     // This is a flex base pointer, but \c SimpleOuter1::fam should not be
///     // included in the analysis in this context since it's the
///     struct SimpleOuter2 *p = malloc(sizeof(struct SimpleOuter2) + len);
///     p->hdr.len = len;
///     return p;
/// }
/// \endcode
///
/// Even if a pointer doesn't directly share a count variable with a FAM,
/// it may belong to the same group indirectly as seen in \c p1 below:
/// \code
/// struct S {
///     int len1;
///     int len2;
///     int * __counted_by(len1) p1;
///     int * __counted_by(len1 + len2) p2;
///     int fam[__counted_by(len2)];
/// };
/// \endcode
void analyzeFlexBase(Sema &S, ValueDecl *VD, Expr *E,
                     AssignedDeclRefResult &Result) {
  std::string FlexBase;
  llvm::raw_string_ostream FlexBaseOS(FlexBase);
  RecordDecl *FlexBaseDecl =
      DynamicCountPointerAssignmentAnalysis::computeFlexBaseKey(E, &FlexBaseOS);
  if (!FlexBaseDecl)
    return;

  // Recursion is needed to check indirect flexbase members, so we need to track
  // which decls we've already visited.
  llvm::SmallPtrSet<const ValueDecl *, 2> Visited;
  std::function<bool(const ValueDecl *)> CheckDependentDecls;
  CheckDependentDecls = [&Visited, &CheckDependentDecls, &FlexBaseDecl,
                         &FlexBase, &Result](const ValueDecl *VD) {
    if (!FlexBaseDecl->isParentStructOf(VD))
      return false; // if this is a nested field referred to by outer structs,
                    // only consider fields in the scope of the relevant struct.
    if (!Visited.insert(VD).second)
      return false;

    if (VD->getType()->isIncompleteArrayType() &&
        VD->getType()->isCountAttributedType()) {
      Result.FlexBaseKey = FlexBase;
      Result.FlexBaseDecl = FlexBaseDecl;
      return true;
    }

    if (auto *Att = VD->getAttr<DependerDeclsAttr>())
      for (const auto *D : Att->dependerDecls())
        if (CheckDependentDecls(cast<ValueDecl>(D)))
          return true;

    const auto *DCPTy = VD->getType()->getAs<CountAttributedType>();
    if (!DCPTy && VD->getType()->isPointerType())
      DCPTy = VD->getType()->getPointeeType()->getAs<CountAttributedType>();

    if (!DCPTy)
      return false;

    for (const auto &DI : DCPTy->dependent_decls())
      if (CheckDependentDecls(cast<ValueDecl>(DI.getDecl())))
        return true;
    return false;
  };

  if (auto *Att = VD->getAttr<DependerDeclsAttr>()) {
    // CheckDependentDecls searches depth first, so check all the direct
    // neighbors first to correctly set the IsCountForFam. If a __counted_by
    // pointer shares count with a FAM it should belong to the same group and
    // thus requires a FlexBaseKey. But it is not the count, and any count
    // variables not shared should not be treated as count for FAM, despite
    // having FlexBaseKey.
    for (const auto *D : Att->dependerDecls()) {
      const auto *DepVD = cast<ValueDecl>(D);
      if (FlexBaseDecl->isParentStructOf(DepVD) &&
          DepVD->getType()->isIncompleteArrayType() &&
          DepVD->getType()->isCountAttributedType()) {
        Result.FlexBaseKey = FlexBase;
        Result.FlexBaseDecl = FlexBaseDecl;
        Result.IsCountForFam = true;
        return;
      }
    }
  }

  // Check if this is a pointer that shares a count with a FAM.
  CheckDependentDecls(VD);
}

/// analyzeAssignedDeclRef - This function analyzes declaration reference in
/// assignment expression, including DeclRefExpr and MemberExpr. This also
/// supports analysis of dereferenced decl like "*p".
void analyzeAssignedDeclRef(Sema &S, Expr *E, AssignedDeclRefResult &Result) {
  E = E->IgnoreParenCasts();
  if (auto UO = dyn_cast<UnaryOperator>(E)) {
    /// This is to support cases like '*&*ptr = ...'.
    switch (UO->getOpcode()) {
    case UO_Deref:
      Result.Level++;
      break;
    case UO_AddrOf:
      ///  '&x' would be non-assignable'.
      if (Result.Level == 0)
        return;
      Result.Level--;
      break;
    default:
      return;
    }
    return analyzeAssignedDeclRef(S, UO->getSubExpr(), Result);
  }

  ValueDecl *VD = nullptr;
  std::string Prefix;
  if (auto ME = dyn_cast<MemberExpr>(E)) {
    llvm::raw_string_ostream PrefixOS(Prefix);
    computeMemberExprKey(ME, PrefixOS);
    VD = ME->getMemberDecl();
    analyzeFlexBase(S, VD, ME, Result);
  } else if (auto DRE = dyn_cast<DeclRefExpr>(E)) {
    VD = DRE->getDecl();
  }
  if (!VD)
    return;

  QualType Ty = getInnerType(E->getType(), Result.Level);
  if (Ty.isNull())
    return;

  Result.Ty = E->getType();
  Result.Loc = E->getExprLoc();
  Result.Key = Prefix;

  analyzeAssignedDeclCommon(S, VD, Ty, Result);
}

bool ReplaceSubExpr(ParentMap &PM, Expr *SubExpr, Expr *NewSubExpr) {
  Stmt *Parent = PM.getParent(SubExpr);
  if (!Parent)
    return false;

  for (auto &C : Parent->children()) {
    if (C == SubExpr) {
      *&C = NewSubExpr;
      PM.setParent(NewSubExpr, Parent);
      PM.setParent(SubExpr, NewSubExpr);
      return true;
    }
  }
  return false;
}

/// Build the bounds check expression for delayed count/range checks.
class PreAssignCheck {
  Sema &SemaRef;
  ParentMap &PM;
  Expr *GuardedValue;

public:
  using DeclToNewValueTy = llvm::DenseMap<const ValueDecl *, std::pair<Expr *, unsigned>>;

private:
  ReplaceDeclRefWithRHS DeclReplacer;

public:
  PreAssignCheck(Sema &SemaRef, ParentMap &PM, Expr *GuardedValue,
                     DeclToNewValueTy &DeclToNewValue)
    : SemaRef(SemaRef),
      PM(PM),
      GuardedValue(GuardedValue),
      DeclReplacer(SemaRef, DeclToNewValue) {}

private:
  ExprResult createZeroValue(Sema &S, QualType Ty) {
    auto &Ctx = SemaRef.Context;
    auto BitWidth = Ctx.getTypeSize(Ctx.IntTy);
    llvm::APInt ZeroVal(BitWidth, 0);
    ExprResult Zero = IntegerLiteral::Create(SemaRef.Context, ZeroVal, Ctx.IntTy,
                                             SourceLocation());
    CastKind CK = CastKind::CK_NoOp;
    auto ConversionType = S.CheckAssignmentConstraints(Ty, Zero, CK, true);
    if (ConversionType == AssignConvertType::Incompatible) {
      // `CK` is not set for this return value so bail out.
      return ExprError();
    }

    switch (CK) {
    case CK_NoOp:
      return Zero.get();
    case CK_IntegralToPointer:
      CK = CK_NullToPointer;
      LLVM_FALLTHROUGH;
    default:
      return S.ImpCastExprToType(Zero.get(), Ty, CK);
    }
  }

public:

  using TypeExprPairTy = std::pair<QualType, Expr *>;

  ExprResult getMaterializedValueIfNot(
      Expr *E, SmallVectorImpl<OpaqueValueExpr *> *MateredExprs = nullptr,
      bool UpdateParent = true) {
    if (isa<OpaqueValueExpr>(E))
      return E;

    if (isa<ImplicitValueInitExpr>(E))
      return createZeroValue(SemaRef, E->getType());

    OpaqueValueExpr *OVE = OpaqueValueExpr::Wrap(SemaRef.Context, E);
    if (MateredExprs) {
      MateredExprs->push_back(OVE);
    }
    if (UpdateParent)
      ReplaceSubExpr(PM, E, OVE);
    return OVE;
  }

  void insertDeclToNewValue(const ValueDecl *VD, Expr *New, unsigned Level = 0) {
    DeclReplacer.DeclToNewValue[VD] = std::make_pair(New, Level);
  }

  void setMemberBase(Expr *Base) {
    DeclReplacer.MemberBase = Base;
  }

  // UpFront -> Materialize something
  // MateredExprsInFlight is not...
  // We need InitListExpr to be in UpFront!
  ExprResult build(SmallVectorImpl<TypeExprPairTy> &Pairs, InitListExpr *IL,
                   SmallVectorImpl<OpaqueValueExpr *> &MateredExprs) {
    Expr *WrappedExpr = IL;
    // Reverse so that the first check is inserted at the outermost.
    for (auto I = Pairs.rbegin(), E = Pairs.rend(); I != E; ++I) {
      auto Pair = *I;
      ExprResult R;
      SmallVectorImpl<OpaqueValueExpr*> *MateredExprsPtr =
          (Pair == Pairs.front()) ? &MateredExprs : nullptr;
      if (const auto *DCPTy = Pair.first->getAs<CountAttributedType>()) {
        R = build(DCPTy, Pair.second, WrappedExpr, MateredExprsPtr);
      } else {
        const auto *DRPTy = Pair.first->getAs<DynamicRangePointerType>();
        assert(DRPTy);
        R = build(DRPTy, Pair.second, WrappedExpr, MateredExprsPtr);
      }
      if (R.isInvalid())
        return ExprError();
      WrappedExpr = R.get();
    }
    return WrappedExpr;
  }

  ExprResult build(QualType Ty, Expr *WidePtr, Expr *WrappedValue = nullptr,
                   SmallVectorImpl<OpaqueValueExpr *> *MateredExprs = nullptr) {
    if (const auto *DCPTy = Ty->getAs<CountAttributedType>()) {
      return build(DCPTy, WidePtr, WrappedValue, MateredExprs);
    } else {
      const auto *DRPTy = Ty->getAs<DynamicRangePointerType>();
      assert(DRPTy);
      return build(DRPTy, WidePtr, WrappedValue, MateredExprs);
    }
  }

  /// This is to build the trap condition necessary for a sized_by or counted_by pointer and its related
  /// count assignments like: Count = NewCount; Ptr = WidePtr;
  ///
  /// The condition to check will look like below:
  /// \code
  /// 0 <= NewCount && (lower(WidePtr) <= WidePtr <= upper(WidePtr)) &&
  /// NewCount <= upper(WidePtr) - WidePtr
  /// \endcode
  /// The reason why we do `NewCount <= upper(WidePtr) - WidePtr` instead of
  /// `WidePtr + NewCount <= upper(WidePtr)` is that the backend optimizes the former better because
  /// for the latter the compiler needs to consider integer overflow on the pointer arithmetic.
  /// For `upper(WidePtr) - WidePtr`, on the other hand, we can assume no overflow is happening because
  /// we already checked `WidePtr <= upper(WidePtr)`.
  ExprResult build(const CountAttributedType *Ty, Expr *WidePtr,
                   Expr *WrappedValue = nullptr,
                   SmallVectorImpl<OpaqueValueExpr *> *MateredExprs = nullptr) {
    if (!WrappedValue)
      WrappedValue = GuardedValue;
    auto &Ctx = SemaRef.Context;

    SmallVector<OpaqueValueExpr *, 2> CommonExprs;
    if (!MateredExprs) {
      MateredExprs = &CommonExprs;
    }

    assert(!Ty->getCountExpr()->HasSideEffects(Ctx));
    ExprResult CountR = DeclReplacer.TransformExpr(Ty->getCountExpr());
    if (CountR.isInvalid())
      return ExprError();
    Expr *Count = CountR.get();
    assert(!Count->HasSideEffects(Ctx));

    SourceLocation Loc = WrappedValue->getBeginLoc();
    // Later, we might be able to merge diagnostics to check if count is
    // non-constant when a pointer is null here instead of doing it separately.
    if (isa<ImplicitValueInitExpr>(WidePtr->IgnoreImpCasts()) ||
        WidePtr->isNullPointerConstantIgnoreCastsAndOVEs(
            SemaRef.getASTContext(), Expr::NPC_NeverValueDependent) !=
            Expr::NPCK_NotNull) {
      if (Ty->isOrNull()) {
        if (MateredExprs->empty())
          return WrappedValue;
        // Materialize the OVEs, but perform the assignment unconditionally.
        // This materialization happens as part of BoundsCheckExpr otherwise.
        auto Result = MaterializeSequenceExpr::Create(
            SemaRef.Context, WrappedValue, *MateredExprs);
        Result = MaterializeSequenceExpr::Create(SemaRef.Context, Result,
                                                 *MateredExprs, true);
        return Result;
      }
      // if wideptr == 0 then count expr should also be 0.
      ExprResult Zero = createZeroValue(SemaRef, Count->getType());
      if (!Zero.get())
        return ExprError();
      ExprResult Cond = SemaRef.CreateBuiltinBinOp(Loc, BO_EQ, Count,
                                                   Zero.get());
      if (Cond.isInvalid())
        return ExprError();
      return SemaRef.BuildBoundsCheckExpr(WrappedValue, Cond.get(), *MateredExprs);
    }

    ExprResult MatWidePtr = getMaterializedValueIfNot(WidePtr, MateredExprs);
    if (!(WidePtr = MatWidePtr.get()))
      return ExprError();

    SmallVector<Expr *, 4> WidePtrSeq;
    ExprResult LowerR = SemaRef.BuildLowerBoundExpr(WidePtr, Loc, Loc, /*RawPointer*/true);
    if (LowerR.isInvalid())
      return ExprError();

    WidePtrSeq.push_back(LowerR.get());
    WidePtrSeq.push_back(WidePtr);

    ExprResult UpperR = SemaRef.BuildUpperBoundExpr(WidePtr, Loc, Loc, /*RawPointer*/true);
    if (UpperR.isInvalid())
      return ExprError();

    WidePtrSeq.push_back(UpperR.get());

    ExprResult WidePtrSeqR = BoundsCheckBuilder::BuildLEChecks(SemaRef,
                                                               WrappedValue->getExprLoc(),
                                                               WidePtrSeq, *MateredExprs);
    if (WidePtrSeqR.isInvalid())
      return ExprError();

    // 0 <= Count
    SmallVector<Expr *, 4> CountCheckSeq;
    if (Count->getType()->isSignedIntegerOrEnumerationType()) {
      auto Zero = createZeroValue(SemaRef, Count->getType());
      if (!Zero.get())
        return ExprError();
      CountCheckSeq.push_back(Zero.get());
    }

    CountCheckSeq.push_back(Count);

    UpperR = SemaRef.BuildUpperBoundExpr(WidePtr, Loc, Loc, /*ToSingle*/true);
    if (UpperR.isInvalid())
      return ExprError();

    ExprResult WidePtrCmpOperand = WidePtr;
    if (Ty->isCountInBytes()) {
      UpperR = CastToCharPointer(SemaRef, UpperR.get());
      WidePtrCmpOperand = CastToCharPointer(SemaRef, WidePtrCmpOperand.get());
      if (WidePtrCmpOperand.isInvalid())
        return ExprError();
    } else {
      // We can't do pointer arithmetic on incomplete pointee types. The
      // `UpperMinusPtr` computation will fail this case and emit a confusing
      // `err_typecheck_arithmetic_incomplete_or_sizeless_type` diagnostic. To
      // avoid this bail out early.
      if (UpperR.get()->getType()->getPointeeType()->isIncompleteType() ||
          WidePtrCmpOperand.get()
              ->getType()
              ->getPointeeType()
              ->isIncompleteType())
        return ExprError();
    }
    // Upper - Ptr
    ExprResult UpperMinusPtr = SemaRef.CreateBuiltinBinOp(Loc, BO_Sub, UpperR.get(),
                                                          WidePtrCmpOperand.get());

    if (UpperMinusPtr.isInvalid())
      return ExprError();

    CountCheckSeq.push_back(UpperMinusPtr.get());
    // 0 <= Count <= (Upper - Ptr)
    ExprResult CountCheckSeqR = BoundsCheckBuilder::BuildLEChecks(SemaRef,
                                                                  Loc,
                                                                  CountCheckSeq, *MateredExprs);
    if (CountCheckSeqR.isInvalid())
      return ExprError();

    if (Ty->isOrNull()) {
      ExprResult NullCheck =
          SemaRef.CreateBuiltinUnaryOp(Loc, UO_LNot, WidePtr);
      if (NullCheck.isInvalid())
        return ExprError();
      // !Ptr || 0 <= Count <= (Upper - Ptr)
      CountCheckSeqR = SemaRef.CreateBuiltinBinOp(Loc, BO_LOr, NullCheck.get(),
                                                  CountCheckSeqR.get());
      if (CountCheckSeqR.isInvalid())
        return ExprError();
    }

    // __counted_by/__sized_by:
    // (Lower <= Ptr <= Upper) && (0 <= Count <= Upper - Ptr)
    // __counted_by_or_null/__sized_by_or_null:
    // (Lower <= Ptr <= Upper) && (!Ptr || (0 <= Count <= Upper - Ptr))
    // This allows null pointers to be assigned to *_or_null variables
    // even if the count is negative or the count is larger than the range of
    // the RHS (null pointers generally have 0 range). Assigning from one
    // *_or_null variable to another works because the RHS is first converted to
    // a 0 range __bidi_indexable if the value is null, so it still passes the
    // range check. We preserve the semantics of __bidi_indexables to
    // trap if OOB when converted to a type that cannot represent the OOB-ness.
    // So the wide pointer (0x0, 0xBAD, 0x00B) will trap when assigned to a
    // *_or_null pointer even though the base pointer is null, because we cannot
    // represent the lower bound in the LHS.
    ExprResult CondSeqR = SemaRef.CreateBuiltinBinOp(
        Loc, BO_LAnd, WidePtrSeqR.get(), CountCheckSeqR.get());
    if (CondSeqR.isInvalid())
      return ExprError();

    return SemaRef.BuildBoundsCheckExpr(WrappedValue, CondSeqR.get(),
                                        *MateredExprs);
  }

  /// This is to build sequential checks, `lower(WidePtr) <= start <= WidePtr <= end <= upper(WidePtr)`.
  ExprResult build(const DynamicRangePointerType *Ty, Expr *WidePtr,
                   Expr *WrappedExpr = nullptr,
                   SmallVectorImpl<OpaqueValueExpr *> *MateredExprs = nullptr) {
    assert(!WidePtr->HasSideEffects(SemaRef.Context) ||
           isa<OpaqueValueExpr>(WidePtr) || WidePtr->containsErrors());
    if (!WrappedExpr)
      WrappedExpr = GuardedValue;
    SmallVector<Expr *, 4> Bounds;
    SmallVector<OpaqueValueExpr *, 4> CommonExprs;
    SourceLocation Loc = WrappedExpr->getBeginLoc();

    if (!MateredExprs)
      MateredExprs = &CommonExprs;
    auto &Ctx = SemaRef.Context;
    if (isa<ImplicitValueInitExpr>(WidePtr->IgnoreImpCasts())) {
      llvm::APInt Zero(Ctx.getTypeSize(WidePtr->getType()), 0);
      WidePtr = IntegerLiteral::Create(SemaRef.Context, Zero, Ctx.getIntPtrType(), Loc);

      ExprResult ImpResult = SemaRef.BuildCStyleCastExpr(
          Loc, Ctx.getTrivialTypeSourceInfo(QualType(Ty, 0)), Loc, WidePtr);
      if (ImpResult.isInvalid())
        return ExprError();
      WidePtr = ImpResult.get();
    }

    ExprResult MatWidePtr = getMaterializedValueIfNot(WidePtr, MateredExprs);
    if (!(WidePtr = MatWidePtr.get()))
      return ExprError();

    auto PushValidOrErr = [&](ExprResult Result) -> bool {
      if (Expr *RE = Result.get()) {
        if (auto *Lit = dyn_cast<IntegerLiteral>(RE)) {
          // This is only possible if TransformExpr transformed an implicit
          // null-to-pointer cast.
          assert(Lit->getValue().isZero());
          Result = SemaRef.ImpCastExprToType(Lit, WidePtr->getType(),
                                             CK_NullToPointer);
          if (!(RE = Result.get()))
            return false;
        }
        Bounds.push_back(RE);
        return true;
      }
      return false;
    };

    if (WidePtr->getType()->isBidiIndexablePointerType()) {
      ExprResult LowerR = SemaRef.BuildLowerBoundExpr(WidePtr, Loc, Loc, /*RawPointer*/true);
      if (LowerR.isInvalid())
        return ExprError();
      Bounds.push_back(LowerR.get());
    }

    if (!PushValidOrErr(WidePtr))
      return ExprError();

    // The pointer that has an associated end pointer is responsible to do the
    // bounds checks:
    //  lb(new_ptr) <= start <= new_ptr  <= end <= ub(new_ptr)
    // Thus, we check if the pointer has an end pointer. This is consistent with
    // how we handle assignments (and this is a routine we handle init lists).
    // XXX: This is based on the assumption that only '__ended_by' is exposed to
    // the users and '__started_by' is an implicit attribute the compiler adds
    // based on the corresponding
    // '__ended_by'. We may revisit this if we decide to expose '__started_by'
    // to users.
    if (auto *End = Ty->getEndPointer()) {
      if (!PushValidOrErr(DeclReplacer.TransformExpr(End)))
        return ExprError();
    }

    ExprResult UpperR = SemaRef.BuildUpperBoundExpr(WidePtr, Loc, Loc, /*RawPointer*/true);
    if (UpperR.isInvalid())
      return ExprError();
    Bounds.push_back(UpperR.get());

    ExprResult Cond =
      BoundsCheckBuilder::BuildLEChecks(SemaRef,
                                        Loc,
                                        Bounds, *MateredExprs);

    if (Cond.isInvalid())
      return ExprError();
    return SemaRef.BuildBoundsCheckExpr(WrappedExpr, Cond.get(), *MateredExprs);
  }

  void buildAndChain(Expr *AssignExpr, Expr *WidePtr) {
    Expr *LHS = nullptr;
    if (auto *BO = dyn_cast<BinaryOperator>(AssignExpr)) {
      LHS = BO->getLHS();
    } else {
      auto *UO = dyn_cast<UnaryOperator>(AssignExpr);
      assert(UO && UO->isIncrementDecrementOp());
      LHS = UO->getSubExpr();
    }

    LHS = LHS->IgnoreParenCasts();
    if (auto *OVE = dyn_cast<OpaqueValueExpr>(LHS)) {
      LHS = OVE->getSourceExpr()->IgnoreParenCasts();
    }

    if (auto *ME = dyn_cast<MemberExpr>(LHS)) {
      setMemberBase(ME->getBase());
    }

    buildAndChain(AssignExpr->getType(), WidePtr);
  }

  void buildAndChain(QualType Ty, Expr *WidePtr) {
    if (!Ty->isBoundsAttributedType())
      return;

    auto *CE = dyn_cast<CastExpr>(WidePtr);
    if (CE && CE->getCastKind() == CK_BoundsSafetyPointerCast)
      WidePtr = CE->getSubExpr();
    ExprResult WidePtrR = ForceRebuild(SemaRef).TransformExpr(WidePtr);
    assert(!WidePtrR.isInvalid());
    WidePtr = WidePtrR.get();

    // This is a good place to modify AssignOp if we need to. Check if any work
    // needs to be done due to dynamic-bound pointer types.
    ExprResult Result;
    if (const auto *DBPTy = Ty->getAs<BoundsAttributedType>()) {
      if (const auto *DCPTy = dyn_cast<CountAttributedType>(DBPTy)) {
        Result = build(DCPTy, WidePtr);
      } else {
        const auto *DRPTy = dyn_cast<DynamicRangePointerType>(DBPTy);
        if (DRPTy->getEndPointer())
          Result = build(DRPTy, WidePtr);
      }
    }

    // The error should have been diagnosed somewhere else.
    if (Result.get())
      GuardedValue = Result.get();
  }

  /// The function generates an assignment check for flexible base similar to
  /// pre-assignment checks on count and pointer pairs.
  ///
  /// @code
  /// f = new_f; // check if `0 <= new_count <= bounds(new_f)`
  /// f->count = new_count;
  /// @endcode
  void buildAndChainFlex(Expr *WidePtr) {
    llvm::SmallVector<OpaqueValueExpr *, 4> OVEs;
    auto *CE = dyn_cast<CastExpr>(WidePtr);
    if (CE && CE->getCastKind() == CK_BoundsSafetyPointerCast &&
        !WidePtr->getType()->isPointerTypeWithBounds())
      WidePtr = CE->getSubExpr();
    Expr *OldWidePtr = WidePtr;
    ExprResult WidePtrR = ForceRebuild(SemaRef).TransformExpr(WidePtr);
    assert(!WidePtrR.isInvalid());
    WidePtr = OpaqueValueExpr::EnsureWrapped(
        SemaRef.Context, WidePtrR.get(), OVEs);

    SourceLocation Loc = GuardedValue->getBeginLoc();

    // This is a good place to modify AssignOp if we need to. Check if any work
    // needs to be done due to dynamic-bound pointer types.
    CopyExpr Copy(SemaRef);
    for (auto It : DeclReplacer.DeclToNewValue)
      Copy.UnsafelyAddDeclSubstitution(const_cast<ValueDecl *>(It.first),
                                       It.second.first->IgnoreImpCasts());
    ExprResult Result =
        BoundsCheckBuilder::CheckFlexibleArrayMemberSizeWithOVEs(
            SemaRef, Loc, BoundsCheckKind::FlexibleArrayCountAssign, WidePtr,
            OVEs, &Copy);
    if (Result.isInvalid())
      return;

    if (!OVEs.empty()) {
      Result = MaterializeSequenceExpr::Create(SemaRef.Context, Result.get(), OVEs);
      Result = MaterializeSequenceExpr::Create(SemaRef.Context, Result.get(), OVEs, true);
    }

    bool Succ = ReplaceSubExpr(PM, OldWidePtr, Result.get());
    (void)Succ;
    assert(Succ);
  }

  std::tuple<Expr *, bool, RecordDecl *> GetFlexBase(Expr *InE) {
    auto *ME = cast<MemberExpr>(InE);
    QualType RecordTy = ME->getBase()->getType();
    if (ME->isArrow())
      RecordTy = RecordTy->getPointeeType();
    auto *RD = RecordTy->getAsRecordDecl();
    assert(RD);
    if (!RD->hasFlexibleArrayMember())
      return GetFlexBase(ME->getBase());
    return {ME->getBase(), ME->isArrow(), RD};
  }

  /// This function generates a code to assert that the new count is
  /// smaller or equal to the original count value.
  ///
  /// @code
  /// typedef struct { int count; char fam[__counted_by(count)]; } flex_t;
  /// flex_t g_flex = ...
  ///
  /// void test() {
  ///   g_flex.count = new_count; // check if `0 <= new_count <= g_flex.count`
  /// }
  /// @endcode
  void buildAndChainOldCountCheck(Expr *AssignExpr) {
    Expr *OldSelf = nullptr;
    if (auto *BO = dyn_cast<BinaryOperator>(AssignExpr)) {
      OldSelf = BO->getLHS();
    } else {
      auto *UO = dyn_cast<UnaryOperator>(AssignExpr);
      assert(UO && UO->isIncrementDecrementOp());
      OldSelf = UO->getSubExpr();
    }
    OldSelf = OldSelf->IgnoreParenCasts();
    if (auto *OVE = dyn_cast<OpaqueValueExpr>(OldSelf))
      OldSelf = OVE->getSourceExpr()->IgnoreParenCasts();
    // Currently, this can only be reached with a member expression,
    // which is the count of flexible array member.

    Expr *Base;
    bool IsArrow;
    RecordDecl *RD;
    // The member expression may be nested, recurse to the flexible array
    // struct.
    std::tie(Base, IsArrow, RD) = GetFlexBase(OldSelf);

    FieldDecl *LastField;
    for (auto *FD : RD->fields())
      LastField = FD;
    assert(LastField->getType()->isIncompleteArrayType());
    const auto *DCPTy = LastField->getType()->getAs<CountAttributedType>();
    assert(DCPTy);
    Expr *Count = DCPTy->getCountExpr();

    SmallVector<OpaqueValueExpr *, 4> CommonExprs;
    ExprResult OpaqueBase;
    if (auto *OVE = dyn_cast<OpaqueValueExpr>(Base)) {
      OpaqueBase = OVE;
    } else {
      OpaqueBase = getMaterializedValueIfNot(Base, &CommonExprs);
      if (OpaqueBase.isInvalid())
        return;
    }

    ExprResult OldCount =
        SemaRef.InstantiateDeclRefField(OpaqueBase.get(), IsArrow, Count);

    SmallVector<Expr *, 3> Counts;
    if (Count->getType()->isSignedIntegerOrEnumerationType()) {
      ExprResult Zero = createZeroValue(SemaRef, Count->getType());
      if (Zero.isInvalid())
        return;
      Counts.push_back(Zero.get());
    }

    ExprResult NewCount = DeclReplacer.TransformExpr(Count);
    NewCount = SemaRef.DefaultLvalueConversion(NewCount.get());
    if (NewCount.isInvalid())
      return;
    Counts.push_back(NewCount.get());

    OldCount = SemaRef.DefaultLvalueConversion(OldCount.get());
    if (OldCount.isInvalid())
      return;
    Counts.push_back(OldCount.get());

    ExprResult LEChecks = BoundsCheckBuilder::BuildLEChecks(
        SemaRef, GuardedValue->getExprLoc(), Counts, CommonExprs);
    // Error should have been handled inside 'BuildLEChecks'.
    if (LEChecks.isInvalid())
      return;
    ExprResult Result =
        SemaRef.BuildBoundsCheckExpr(GuardedValue, LEChecks.get(), CommonExprs);
    if (Result.isInvalid())
      return;
    GuardedValue = Result.get();
  }

  /// Perform if the new assignment is within the old range when the rest of the dependent pointers will not be changed.
  bool buildAndChainOldRangeCheck(Expr *AssignExpr, Expr *NewSelf) {
    const auto *DRPTy = AssignExpr->getType()->getAs<DynamicRangePointerType>();
    assert(DRPTy && (DRPTy->getStartPointer() || DRPTy->getEndPointer()));
    SmallVector<Expr *, 4> Ptrs;
    Expr *OldSelf = nullptr;
    if (auto *BO = dyn_cast<BinaryOperator>(AssignExpr)) {
      OldSelf = BO->getLHS();
    } else {
      auto *UO = dyn_cast<UnaryOperator>(AssignExpr);
      assert(UO && UO->isIncrementDecrementOp());
      OldSelf = UO->getSubExpr();
    }

    // The new pointer might be still within the old range, but the buffer has possibly
    // been resized its bounds, e.g., by calling realloc(). For such cases, we fallback
    // to generic range checks.
    if (NewSelf->HasSideEffects(SemaRef.Context))
      return false;

    // old_start <= new_ptr_val <= old_end
    auto rebuildAndPushPtr = [&](Expr *RangePtr, Expr *OldSelf = nullptr) {
      if (!RangePtr) {
        assert(OldSelf && !OldSelf->HasSideEffects(SemaRef.Context));
        ExprResult OldSelfR = ForceRebuild(SemaRef).TransformExpr(OldSelf);
        OldSelfR = SemaRef.DefaultFunctionArrayLvalueConversion(OldSelfR.get());
        assert(!OldSelfR.isInvalid());
        Ptrs.push_back(OldSelfR.get());
        return;
      }

      assert(!RangePtr->HasSideEffects(SemaRef.Context));
      // Since we materialize self assignments, we can reuse the materialized value.
      ExprResult NewRangePtr = DeclReplacer.TransformExpr(RangePtr);
      assert(!NewRangePtr.isInvalid());
      // This is assuming the range has not been changed.
      NewRangePtr = SemaRef.DefaultFunctionArrayLvalueConversion(NewRangePtr.get());
      assert(!NewRangePtr.isInvalid());
      Ptrs.push_back(NewRangePtr.get());
    };

    rebuildAndPushPtr(DRPTy->getStartPointer(), OldSelf);
    rebuildAndPushPtr(nullptr, NewSelf);
    rebuildAndPushPtr(DRPTy->getEndPointer(), OldSelf);

    SmallVector<OpaqueValueExpr *, 4> CommonExprs;
    ExprResult LEChecks = BoundsCheckBuilder::BuildLEChecks(SemaRef,
                                                            GuardedValue->getExprLoc(),
                                                            Ptrs, CommonExprs);
    assert(!LEChecks.isInvalid());
    ExprResult Result = SemaRef.BuildBoundsCheckExpr(GuardedValue,
                                                     LEChecks.get(),
                                                     CommonExprs);
    assert(!Result.isInvalid());
    GuardedValue = Result.get();
    return true;
  }

  Expr *getChainedBoundsCheckExpr() const {
    return GuardedValue;
  }

};

class DeclRefFinder : public ConstEvaluatedExprVisitor<DeclRefFinder> {
public:
  typedef ConstEvaluatedExprVisitor<DeclRefFinder> Inherited;
  const Decl *DeclToFind;
  const Expr *Result;

  DeclRefFinder(ASTContext &Ctx, const Expr *Haystack, const Decl *Needle)
      : Inherited(Ctx), DeclToFind(Needle) {
    Visit(Haystack);
  }

  const Expr *getExpr() { return Result; }

  void VisitMemberExpr(const MemberExpr *E) {
    if (E->getMemberDecl() == DeclToFind) {
      Result = E;
    }
  }

  void VisitDeclRefExpr(const DeclRefExpr *E) {
    if (E->getDecl() == DeclToFind) {
      Result = E;
    }
  }
};

class CountDepGroup;
class RangeDepGroup;

/// Group decls with the decl referenced in count expression as a key decl.
/// Pointers that share the same count decl are in the same group. We currently
/// have "a" key decl because we don't currently allow having multiple count
/// variables referenced in a count expression.
/// FIXME: This should be extended to have multiple count declarations.
/// rdar://70692513
class DepGroup {
public:
  enum DepGroupKind { DGK_Count, DGK_Range };
private:
  const DepGroupKind Kind;

protected:
  class AssignExprInfo {
    llvm::PointerIntPair<Expr *, 2> Data;
    llvm::PointerIntPair<ValueDecl *, 2> ExtData;
    // The map 'ReferencedDecls' tracks dependent Decls referenced inside the assignment
    // expression and the nested level of the decl reference, for the sake of diagnostics.
    llvm::DenseMap<const ValueDecl *, unsigned> ReferencedDecls;
    enum {
      SELF_ASSIGN = 0,
      COUNT_VAR = 1,
      COUNT_FOR_FAM = 0,
      FLEX_BASE = 1,
    };

  public:
    explicit AssignExprInfo(Expr *E, ValueDecl *VD, bool SelfAssign,
                            bool CountVar, bool CountForFam, bool FlexBase)
        : Data(E, ((SelfAssign ? 1 << SELF_ASSIGN : 0) |
                   (CountVar ? 1 << COUNT_VAR : 0))),
          ExtData(VD, ((CountForFam ? 1 << COUNT_FOR_FAM : 0) |
                       (FlexBase ? 1 << FLEX_BASE : 0))) {}

    bool IsSelfAssign() const { return (Data.getInt() >> SELF_ASSIGN) & 1; }
    bool IsCountVar() const { return (Data.getInt() >> COUNT_VAR) & 1; }
    bool IsCountForFam() const {
      return (ExtData.getInt() >> COUNT_FOR_FAM) & 1;
    }
    bool IsFlexBase() const { return (ExtData.getInt() >> FLEX_BASE) & 1; }
    void AddReferencedDecl(const ValueDecl *VD, unsigned Level) {
      ReferencedDecls.insert(std::make_pair(VD, Level));
    }
    const llvm::DenseMap<const ValueDecl*, unsigned>& getReferencedDecls() const {
      return ReferencedDecls;
    }

    bool IsDeclInit() const { return !getExpr() && getDeclInit(); }
    ValueDecl *getValueDecl() const { return ExtData.getPointer(); }
    Expr *getExpr() const { return Data.getPointer(); }
    Expr *getDeclInit() const {
      auto Var = dyn_cast<VarDecl>(getValueDecl());
      if (!Var)
        return nullptr;
      return Var->getInit();
    }
    Expr *getAssignOrInitExpr() const {
      return IsDeclInit() ? getDeclInit() : getExpr();
    }
  };

protected:
  bool SideEffectAfter = false;
  bool SkipFlexCheck = false;
  bool FlexBaseNull = false;
  std::string Key;
  std::string FlexBaseKey;
  /// FIXME: Support multiple dependent lengths in a length expression.
  llvm::SmallPtrSet<const ValueDecl *, 2> KeyDecls;
  // The map 'AssignDecls' keeps a link to the expression that assigns to
  // the decl for diagnostics, i.e., the 'unsigned' data is an index to
  // 'AssignExprList'.
  llvm::DenseMap<const ValueDecl *, unsigned> AssignedDecls;
  llvm::DenseMap<const ValueDecl *, std::pair<Expr *, unsigned>> DeclToNewValue;
  llvm::SmallPtrSet<const ValueDecl *, 1> SelfAssignedDecls;
  llvm::SmallVector<AssignExprInfo, 1> AssignExprList;
  llvm::SmallVector<OpaqueValueExpr *, 4> MateredExprsUpFront;

  std::string getPrefixedName(const ValueDecl *VD, unsigned NestedLevel = 0) const {
    std::string starPrefix = "";
    while (NestedLevel--)
      starPrefix += "*";
    return (starPrefix + Key + VD->getName()).str();
  }

  std::string getPrefixedNameWithFlexBase(const Expr *E,
                                          unsigned NestedLevel = 0) const {
    std::string starPrefix = "";
    while (NestedLevel--)
      starPrefix += "*";
    std::string MemberExprString;
    llvm::raw_string_ostream MemberExprStringOS(MemberExprString);
    computeMemberExprKey(E, MemberExprStringOS, /*Top*/ false);
    return starPrefix + FlexBaseKey + MemberExprString;
  }

  bool isFlexBasePointer() const {
    size_t KeyLen = FlexBaseKey.length();
    return KeyLen > 2 && FlexBaseKey.substr(KeyLen - 2, 2) == "->";
  }

  bool hasAssignToFlexBase() const {
    return AssignExprList.back().IsFlexBase();
  }

  bool hasFlexBase() const { return !FlexBaseKey.empty(); }

  std::string getFlexBaseName() const {
    if (FlexBaseKey.empty())
      return "";
    size_t KeyLen = FlexBaseKey.length();
    if (isFlexBasePointer())
      return FlexBaseKey.substr(0, KeyLen - 2);
    assert(KeyLen > 1 && FlexBaseKey.substr(KeyLen - 1, 1) == ".");
    return FlexBaseKey.substr(0, KeyLen - 1);
  }

  void FinalizePreAssignCheck(Sema &SemaRef, ParentMap &PM, Expr *LastBoundsCheckExpr);

public:
  explicit DepGroup(DepGroupKind Kind, StringRef Key, StringRef FlexBaseKey)
      : Kind(Kind), Key(Key), FlexBaseKey(FlexBaseKey) {
    // Produce KeyDecls
  }

  virtual ~DepGroup() {}

  void Finalize(Sema &S, ParentMap &PM);

  virtual void EmitChecksToAST(Sema &SemaRef, ParentMap &PM) = 0;

  DepGroupKind getKind() const { return Kind; }

  Expr *getAssignExpr(size_t Index) const {
    if (Index >= AssignExprList.size())
      return nullptr;
    return AssignExprList[Index].getExpr();
  }

  void setSideEffectAfter() { SideEffectAfter = true; }
  void clearSideEffectAfter() { SideEffectAfter = false; }
  bool hasSideEffectAfter() const { return SideEffectAfter; }
  void setSkipFlexCheck() { SkipFlexCheck = true; }
  bool skipFlexCheck() const { return SkipFlexCheck; }
  void setFlexBaseNull() { FlexBaseNull = true; }
  bool isFlexBaseNull() const { return FlexBaseNull; }

  bool insertAssignedDecl(const ValueDecl *D, bool IsSelfAssign = false) {
    auto It = AssignedDecls.find(D);
    if (It != AssignedDecls.end())
      return false;
    AssignedDecls[D] = AssignExprList.size();
    if (IsSelfAssign)
      SelfAssignedDecls.insert(D);
    return true;
  }
  bool insertDeclToNewValue(const ValueDecl *D, Expr *NewValue, unsigned Level) {
    auto It = DeclToNewValue.find(D);
    if (It != DeclToNewValue.end())
      return false;
    DeclToNewValue[D] = std::make_pair(NewValue, Level);
    return true;
  }

  // This is in reverse order.
  void insertMateredExpr(OpaqueValueExpr *OVE) {
    MateredExprsUpFront.push_back(OVE);
  }

  void insertMateredExprsReverse(llvm::SmallVectorImpl<OpaqueValueExpr *> &OVEs) {
    std::reverse(OVEs.begin(), OVEs.end());
    MateredExprsUpFront.append(OVEs);
  }

  bool hasSelfAssign() const { return !SelfAssignedDecls.empty(); }
  void addReferencedDecl(const ValueDecl *VD, unsigned Level) {
    assert(!AssignExprList.empty());
    assert(!AssignExprList.back().IsSelfAssign());
    AssignExprList.back().AddReferencedDecl(VD, Level);
  }
  bool isDeclInitBasedGroup() const { return AssignExprList.empty(); }
  void updateLastAssign(Expr *E, ValueDecl *VD, bool SelfAssign, bool CountVar,
                        bool CountForFam, bool FlexBase) {
    AssignExprList.emplace_back(E, VD, SelfAssign, CountVar, CountForFam,
                                FlexBase);
  }
  Expr *getLastAssign() {
    return AssignExprList.empty() ? nullptr
                                  : AssignExprList[0].getAssignOrInitExpr();
  }
  Expr *getFirstAssign() {
    return AssignExprList.empty()
               ? nullptr
               : AssignExprList.rbegin()->getAssignOrInitExpr();
  }

  const AssignExprInfo &getFirstAssignExprInfoUnsafe() const {
    return *AssignExprList.rbegin();
  }

  virtual bool DiagnoseMissingDependentAssign(Sema &SemaRef, const ValueDecl *VD) const = 0;
  bool matchKey(StringRef K, const ValueDecl *VD) const {
    return Key == K && KeyDecls.find(VD) != KeyDecls.end();
  }
  bool matchFlexBaseKey(StringRef K) const {
    return !FlexBaseKey.empty() && FlexBaseKey == K;
  }
  const Expr *getAssignExprForDecl(const ValueDecl *D) const {
    auto It = AssignedDecls.find(D);
    if (It == AssignedDecls.end() || It->second >= AssignExprList.size())
      return nullptr;
    return AssignExprList[It->second].getExpr();
  }

  bool CheckDynamicCountAssignments(Sema &SemaRef);
};

void DepGroup::FinalizePreAssignCheck(Sema &SemaRef,
                                          ParentMap &PM,
                                          Expr *LastBoundsCheckExpr) {
  if (!MateredExprsUpFront.empty()) {
    std::reverse(MateredExprsUpFront.begin(), MateredExprsUpFront.end());
    bool FirstIsLast = (getFirstAssign() == getLastAssign());
    assert(getLastAssign());
    // FIXME: Binding comes first in the other places
    Expr *UnbindExpr = FirstIsLast ? LastBoundsCheckExpr : getLastAssign();
    UnbindExpr = MaterializeSequenceExpr::Create(SemaRef.Context,
                                                 UnbindExpr,
                                                 MateredExprsUpFront,
                                                 /*Unbind*/true);
    Expr *BindExpr = FirstIsLast ? UnbindExpr : LastBoundsCheckExpr;
    BindExpr = MaterializeSequenceExpr::Create(SemaRef.Context,
                                               BindExpr,
                                               MateredExprsUpFront);
    LastBoundsCheckExpr = BindExpr;
    if (!FirstIsLast) {
      bool Succ = ReplaceSubExpr(PM, getLastAssign(), UnbindExpr);
      (void)Succ;
      assert(Succ);
    }
  }

  if (getFirstAssign() != LastBoundsCheckExpr) {
    bool Succ = ReplaceSubExpr(PM, getFirstAssign(), LastBoundsCheckExpr);
    (void)Succ;
    assert(Succ);
  }
}

static bool IsModifiableValue(Sema &S, const ValueDecl *VD) {
  if (clang::IsConstOrLateConst(VD))
    return false;
  Expr *TempLValueForVD = S.BuildDeclRefExpr(
      const_cast<ValueDecl *>(VD), VD->getType(), VK_LValue, SourceLocation());
  return TempLValueForVD->isModifiableLvalue(S.Context) == Expr::MLV_Valid;
}

/// Finalize closes the current decl group analysis window. Before closing,
/// the function diagnoses the analyzed group to check if there was any missing
/// assignment to declaration within the same group.
void DepGroup::Finalize(Sema &SemaRef, ParentMap &PM) {
  bool HadError = false;
  for (const auto *Var : KeyDecls) {
    /// We don't worry about instantiation here because we've already checked
    /// the base as part of the group key.
    if (AssignedDecls.find(Var) == AssignedDecls.end() &&
        IsModifiableValue(SemaRef, Var)) {
      /// FIXME: Improve diagnostics using the base of field
      SourceLocation Loc;
      SourceRange Range;
      /// We can have empty AssignExprList since we also track DeclStmt.
      if (!AssignExprList.empty()) {
        /// Continue if there was no directly dependent assignment to report.
        /// In the following example,
        /// we don't report on missing assignment to 'ptr2' and only report on
        /// 'len' on which 'ptr' immediately depends.
        /// \code
        /// struct Foo {
        ///  int len;
        ///  int *__counted_by(len) ptr;
        ///  int *__counted_by(len) ptr2;
        /// };
        /// struct Foo f;
        /// f.ptr = nullptr;
        /// \endcode
        if (!DiagnoseMissingDependentAssign(SemaRef, Var))
          continue;
      } else {
        Loc = Var->getLocation();
        Range = Var->getSourceRange();
        SemaRef.Diag(Loc, diag::err_bounds_safety_non_adjacent_dependent_var_decl)
            << getPrefixedName(Var) << Range;
      }
      HadError = true;
    }
  }

  /// This diagnoses if the consecutive assignments are ordered in a way to introduce
  /// an unexpected runtime trap, as in the following example.
  /// \code
  /// struct S{
  ///   unsigned len;
  ///   int *__counted_by(len) ptr1;
  ///   int *__counted_by(len) ptr2;
  /// };
  /// void foo(struct S *s) {
  ///   s->ptr1 = s->ptr1 + 1; // ok: 's->ptr1' is referenced before updated.
  ///   s->ptr2 = s->ptr1; // error: cannot reference 's->ptr1' after it is changed during ...
  ///   s->len = 10;
  /// }
  /// \endcode
  /// The following routine checks if a decl referenced in the assignment has an implicitly
  /// dependent decl (e.g., 'ptr' is dependent to 'len' in the above example). If so, it
  /// diagnoses if an assignment to the dependent decl (e.g., 'len') preceeds the current
  /// assignment in the analyzed group. The same applies to '__ended_by'.
  bool HasHadCountForFamError = false;
  for (unsigned i = AssignExprList.size(); i--;) {
    auto &ExprInfo = AssignExprList[i];
    if (ExprInfo.IsSelfAssign())
      continue;

    /// We skip bounds checks for single to single with flexible array member as
    /// any other single to single cast. However, we still prevent the count
    /// being silently unchecked by requiring the base to be initialized with a
    /// wide pointer.
    /// \code
    /// flex = new_single_flex;
    /// flex->count = new_count;
    /// \endcode
    if (ExprInfo.IsCountForFam() &&
        (!hasAssignToFlexBase() || (skipFlexCheck() && !isFlexBaseNull())) &&
        isFlexBasePointer() && !HasHadCountForFamError) {
      const Expr *AssignedExpr = nullptr;
      if (auto AssignOp = dyn_cast<BinaryOperator>(ExprInfo.getExpr())) {
        assert(AssignOp->getOpcode() == BO_Assign);
        AssignedExpr = AssignOp->getLHS();
      } else if (auto UOp = dyn_cast<UnaryOperator>(ExprInfo.getExpr())) {
        AssignedExpr = UOp->getSubExpr();
      }
      assert(AssignedExpr);
      std::string AssignedExprString;
      llvm::raw_string_ostream AssignedExprStringOS(AssignedExprString);
      computeMemberExprKey(AssignedExpr, AssignedExprStringOS, /*Top*/ false);
      const bool IsFirstNonFlexBase =
          i == AssignExprList.size() - 1 ||
          (hasAssignToFlexBase() && i == AssignExprList.size() - 2);
      SemaRef.Diag(ExprInfo.getExpr()->getExprLoc(),
                   diag::err_bounds_safety_no_preceding_fam_base_assignment)
          << AssignedExprString << getFlexBaseName() << IsFirstNonFlexBase;
      if (skipFlexCheck() && hasAssignToFlexBase()) {
        SemaRef.Diag(getFirstAssign()->getExprLoc(),
                     diag::note_bounds_safety_flex_base_assign)
            << getFlexBaseName();
      } else if (i < AssignExprList.size() - 1) {
        SemaRef.Diag(getFirstAssign()->getExprLoc(),
                     diag::note_bounds_safety_first_dep_assign)
            << getFlexBaseName();
      }
      HadError = HasHadCountForFamError = true;
    }

    for (auto It : ExprInfo.getReferencedDecls()) {
      const auto *RefDecl = cast<ValueDecl>(It.first);

      auto RefUpdateIt = AssignedDecls.find(RefDecl);
      if (RefUpdateIt != AssignedDecls.end() && RefUpdateIt->second > i) {
        auto *RefUpdateExpr = getAssignExprForDecl(RefDecl);
        // Referenced after updated.
        SemaRef.Diag(ExprInfo.getExpr()->getExprLoc(),
                     diag::err_bounds_safety_dependent_assignments_order)
            << getPrefixedName(RefDecl);
        SemaRef.Diag(RefUpdateExpr->getExprLoc(), diag::note_bounds_safety_decl_assignment)
            << getPrefixedName(RefDecl);
        HadError = true;
      }
    }
  }

  HadError |= CheckDynamicCountAssignments(SemaRef);

  if (HadError)
    return;

  EmitChecksToAST(SemaRef, PM);
}

// Check assignment to dynamic count pointer. Use the computed dependent values
// within the assignemnt group and pass them to
// CheckDynamicCountSizeForAssignment() to check all constraints.
bool DepGroup::CheckDynamicCountAssignments(Sema &SemaRef) {
  bool HadError = false;

  for (const auto &ExprInfo : AssignExprList) {
    const auto *BinOp =
        dyn_cast<BinaryOperator>(ExprInfo.getAssignOrInitExpr());
    if (!BinOp || BinOp->getOpcode() != BO_Assign)
      continue;

    const Expr *LHS = BinOp->getLHS();
    QualType LHSTy = LHS->getType();
    const auto *DCPT = LHSTy->getAs<CountAttributedType>();
    if (!DCPT)
      continue;

    Expr *RHS = BinOp->getRHS();

    // Ignore Parens, ImpCasts and OVEs in RHS.
    auto IgnoreOVE = [](Expr *E) -> Expr * {
      if (auto *OVE = dyn_cast<OpaqueValueExpr>(E))
        return OVE->getSourceExpr();
      return E;
    };
    RHS = IgnoreExprNodes(RHS, IgnoreParensSingleStep,
                          IgnoreImplicitCastsSingleStep, IgnoreOVE);

    const ValueDecl *VD = nullptr;
    if (const auto *DRE = dyn_cast<DeclRefExpr>(LHS))
      VD = DRE->getDecl();

    Expr *LHSMemberBase = nullptr;
    if (auto *ME = dyn_cast<MemberExpr>(LHS))
      LHSMemberBase = ME->getBase();

    if (!SemaRef.CheckDynamicCountSizeForAssignment(
            LHSTy, RHS, AssignmentAction::Assigning, BinOp->getExprLoc(),
            VD ? VD->getName() : StringRef(), DeclToNewValue, LHSMemberBase)) {
      HadError = true;
    }
  }

  return HadError;
}

class CountDepGroup : public DepGroup {
protected:
  static ExprResult MakeWidePtrFromDecl(Sema &SemaRef, ValueDecl *VD,
                                        SourceLocation Loc) {
    auto *DRE = SemaRef.BuildDeclRefExpr(VD, VD->getType(), VK_LValue, Loc);
    return SemaRef.DefaultLvalueConversion(DRE);
  }

  void EmitChecksToAST(Sema &SemaRef, ParentMap &PM) override {
    auto *const FA = getFirstAssign();
    if (!FA)
      return;
    if (const auto *AE = dyn_cast<BinaryOperator>(FA)) {
      assert(BinaryOperator::isAssignmentOp(AE->getOpcode()));
      if (SemaRef.allowBoundsUnsafePointerAssignment(
              AE->getLHS()->getType(), AE->getRHS(), AE->getExprLoc()))
        return;
    }
    PreAssignCheck Builder(SemaRef, PM, FA, DeclToNewValue);
    /// We insert information on delayed or non-delayed checks for dynamic count
    /// assignments.
    /// BoundsSafety mandates the assignments to decl within a group should be all
    /// or nothing which means the user needs to add a self assignment for
    /// unchanged variables. If there was a self assignment to a dynamic count
    /// pointer, the result of count expression with the new count can't be
    /// bigger than the result with the old count since the bound of the pointer
    /// hasn't changed.
    if (hasSelfAssign() || hasFlexBase()) {
      for (const auto &ExprInfo : AssignExprList) {
        if (!ExprInfo.IsCountVar() || ExprInfo.IsSelfAssign())
          continue;
        const Expr *E = ExprInfo.getExpr()->IgnoreParenImpCasts();
        const auto *UO = dyn_cast<UnaryOperator>(E);
        assert(!UO || UO->isIncrementDecrementOp());
        if (!UO || !UO->isIncrementOp())
          continue;

        const ValueDecl *VD = ExprInfo.getValueDecl();
        auto *Attr = VD->getAttr<DependerDeclsAttr>();
        for (unsigned i = 0; i < Attr->dependerDecls_size(); ++i) {
          unsigned DepLevel = Attr->dependerLevels_begin()[i];
          auto *DepVD = cast<ValueDecl>(Attr->dependerDecls_begin()[i]);
          auto It = AssignedDecls.find(DepVD);
          bool NoDepAssign = It == AssignedDecls.end();
          assert(!NoDepAssign || ExprInfo.IsCountForFam());
          if (NoDepAssign) {
            SemaRef.Diag(E->getExprLoc(),
                         diag::err_bounds_safety_increment_dynamic_count_nodep)
                << getPrefixedName(VD, Attr->getIsDeref());
          } else if (AssignExprList[It->second].IsSelfAssign()) {
            SemaRef.Diag(E->getExprLoc(),
                         diag::err_bounds_safety_increment_dynamic_count)
                << getPrefixedName(VD, Attr->getIsDeref())
                << getPrefixedName(DepVD, DepLevel);
            return;
          }
        }
      }
    }

    // For flexible base as, having at least two expressions in the group
    // means it has a member assignment.
    auto *MemberAccess = getAssignExpr(AssignExprList.size() - 2);
    if (MemberAccess && isFlexBaseNull()) {
      assert(skipFlexCheck());
      // Report error as there is a subsequent member access on this NULL
      // base.
      SemaRef.Diag(MemberAccess->getExprLoc(),
                   diag::err_bounds_safety_member_reference_null_base);
      return;
    }

    if (SemaRef.getDiagnostics().hasUnrecoverableErrorOccurred())
      return;

    const auto &FirstExprInfo = getFirstAssignExprInfoUnsafe();
    if (FirstExprInfo.IsFlexBase()) {
      if (!skipFlexCheck()) {
        Builder.buildAndChainFlex(
            DeclToNewValue[FirstExprInfo.getValueDecl()].first);
      }
    } else if (FirstExprInfo.IsCountForFam()) {
      Builder.buildAndChainOldCountCheck(FirstExprInfo.getExpr());
    }
    for (const auto &ExprInfo : AssignExprList) {
      if (ExprInfo.IsFlexBase() || ExprInfo.IsCountForFam())
        continue;
      const auto *Expr = ExprInfo.getExpr();
      const auto *VD = ExprInfo.getValueDecl();
      Builder.buildAndChain(ExprInfo.getExpr(),
                            DeclToNewValue[ExprInfo.getValueDecl()].first);
      if (isValueDeclOutCount(VD)) {
        const auto *Att = VD->getAttr<DependerDeclsAttr>();
        for (Decl *DepD : Att->dependerDecls()) {
          auto *DepVD = dyn_cast<ValueDecl>(DepD);
          if (!DepVD || isValueDeclOutBuf(DepVD))
            continue;
          if (DeclToNewValue.find(DepVD) != DeclToNewValue.end())
            continue;
          auto WidePtrR =
              MakeWidePtrFromDecl(SemaRef, DepVD, Expr->getBeginLoc());
          if (!WidePtrR.get())
            return;
          auto *OVE = OpaqueValueExpr::Wrap(SemaRef.Context, WidePtrR.get());
          insertMateredExpr(OVE);
          Builder.buildAndChain(DepVD->getType(), OVE);
        }
      }
    }
    FinalizePreAssignCheck(SemaRef, PM, Builder.getChainedBoundsCheckExpr());
  }

  bool DiagnoseMissingDependentAssign(Sema &SemaRef, const ValueDecl *VD) const override {
    if (!VD->getType()->isIntegralOrEnumerationType() &&
        !VD->getType()->isPointerType())
      return false;
    if (auto *Att = VD->getAttr<DependerDeclsAttr>()) {
      unsigned FirstAssignIndex = 0;
      bool HasDepAssign = false;
      const ValueDecl *AssignedDecl = nullptr;
      unsigned PtrLevel = 0;
      for (unsigned i = 0; i < Att->dependerDecls_size(); ++i) {
        const Decl *D = Att->dependerDecls_begin()[i];
        assert(isa<ValueDecl>(D));
        const ValueDecl *KeyVD = cast<ValueDecl>(D);

        if (KeyVD->getType()->isIncompleteArrayType() &&
            hasAssignToFlexBase()) {
          // This must be flexible array member and `VD` must be referred
          // to by the count expression for fam.
          const auto *DCPTy = KeyVD->getType()->getAs<CountAttributedType>();
          assert(DCPTy);
          bool HasFlexDepAssign = false;
          unsigned FirstFlexAssignIndex = 0;
          const ValueDecl *FlexAssignedDecl = nullptr;
          unsigned FlexPtrLevel = 0;
          for (const auto &DI : DCPTy->getCoupledDecls()) {
            const auto *D = DI.getDecl();
            if (D == VD)
              continue;
            assert(isa<ValueDecl>(D));
            const auto *DepVD = cast<ValueDecl>(D);
            auto It = AssignedDecls.find(DepVD);
            // Report the missing assignment error only if there is an
            // assignment to any other decl referred to by the same count
            // expression of fam.
            // Find the first assignment (the bigger the index, the earlier the statement is)
            // that creates a direct dependency with 'VD'.
            if (It != AssignedDecls.end() &&
                // Don't report missing assignments for structs containing a
                // referenced field.
                DepVD->getType()->isIntegralOrEnumerationType() &&
                (!HasFlexDepAssign || FirstFlexAssignIndex < It->second)) {
              HasFlexDepAssign = true;
              FirstFlexAssignIndex = It->second;
              FlexAssignedDecl = DepVD;
              FlexPtrLevel = DI.isDeref();
            }
          }
          if (HasFlexDepAssign) {
            const auto &ExprInfo = AssignExprList[FirstFlexAssignIndex];
            const Expr *DependentAssign = ExprInfo.getExpr();
            QualType DependerType = getInnerType(FlexAssignedDecl->getType(), FlexPtrLevel);
            DeclRefFinder VDFinder(SemaRef.getASTContext(),
                                   DCPTy->getCountExpr(), VD);
            const Expr *VDExpr = VDFinder.getExpr();
            assert(VDExpr);
            SemaRef.Diag(DependentAssign->getExprLoc(),
                         diag::err_bounds_safety_missing_dependent_var_assignment)
                << getPrefixedName(FlexAssignedDecl, FlexPtrLevel)
                << getPrefixedNameWithFlexBase(VDExpr, Att->getIsDeref())
                << DependerType << 0;
            return true;
          }
        } else {
          auto It = AssignedDecls.find(KeyVD);
          // Find the first assignment (the bigger the index, the earlier the statement is)
          // that creates a direct dependency with 'VD'.
          if (It != AssignedDecls.end() &&
              (!HasDepAssign || FirstAssignIndex < It->second)) {
            HasDepAssign = true;
            FirstAssignIndex = It->second;
            AssignedDecl = KeyVD;
            PtrLevel = Att->dependerLevels_begin()[i];
          }
        }
      }
      if (HasDepAssign) {
        const auto &ExprInfo = AssignExprList[FirstAssignIndex];
        const Expr *DependentAssign = ExprInfo.getExpr();
        QualType DependerType = getInnerType(AssignedDecl->getType(), PtrLevel);

        if (!SemaRef.allowBoundsUnsafeAssignment(DependentAssign->getExprLoc()))
          SemaRef.Diag(DependentAssign->getExprLoc(),
                       diag::err_bounds_safety_missing_dependent_var_assignment)
              << getPrefixedName(AssignedDecl, PtrLevel)
              << getPrefixedName(VD, Att->getIsDeref()) << DependerType << 0;
      }
      return HasDepAssign;
    }
    const auto *DCPTy = VD->getType()->getAs<CountAttributedType>();
    unsigned PtrLevel = 0;
    if (!DCPTy && VD->getType()->isPointerType()) {
      DCPTy = VD->getType()->getPointeeType()->getAs<CountAttributedType>();
      PtrLevel = 1;
    }
    if (!DCPTy)
      return false;
    assert(DCPTy);
    unsigned FirstAssignIndex = 0;
    bool HasDepAssign = false;
    const ValueDecl *AssignedDecl = nullptr;
    unsigned AssigneeLevel = 0;
    // We don't force the user to assign VD if VD is non-inout buf parameter and
    // all dependent declarations are inout counts.
    bool CanSkipAssign = isa<ParmVarDecl>(VD) && !isValueDeclOutBuf(VD);
    for (const auto &DI : DCPTy->dependent_decls()) {
      const auto *D = DI.getDecl();
      assert(isa<ValueDecl>(D));
      const ValueDecl *KeyVD = cast<ValueDecl>(D);
      auto It = AssignedDecls.find(KeyVD);
      if (It != AssignedDecls.end() &&
          (!HasDepAssign || FirstAssignIndex < It->second)) {
        HasDepAssign = true;
        FirstAssignIndex = It->second;
        AssignedDecl = KeyVD;
        AssigneeLevel = DI.isDeref();
      }
      CanSkipAssign &= isValueDeclOutCount(D);
    }
    bool Diagnose = HasDepAssign && !CanSkipAssign;
    if (Diagnose) {
      const auto &ExprInfo = AssignExprList[FirstAssignIndex];
      const Expr *DependentAssign = ExprInfo.getExpr();
      if (!SemaRef.allowBoundsUnsafeAssignment(DependentAssign->getExprLoc()))
        SemaRef.Diag(DependentAssign->getExprLoc(),
                     diag::err_bounds_safety_missing_dependent_var_assignment)
            << getPrefixedName(AssignedDecl, AssigneeLevel)
            << getPrefixedName(VD, PtrLevel) << QualType(DCPTy, 0) << 1;
    }
    return Diagnose;
  }

public:
  explicit CountDepGroup(const ValueDecl *VD, StringRef Key,
                         StringRef FlexBaseKey, const RecordDecl *FlexBaseDecl)
      : DepGroup(DGK_Count, Key, FlexBaseKey) {
    std::function<void(const ValueDecl*)> AddDependentDecls;
    AddDependentDecls = [this, &AddDependentDecls,
                         &FlexBaseDecl](const ValueDecl *VD) {
      if (FlexBaseDecl && !FlexBaseDecl->isParentStructOf(VD))
        return; // if this is a nested field referred to by outer structs,
                // only consider fields in the scope of the relevant struct.
      if (!KeyDecls.insert(VD).second)
        return;

      // Skip adding decls that use a const/late const. E.g.:
      //
      // ```
      // extern const unsigned extern_const_global_count;
      //
      // void fun_extern_const_global_count(
      //    int *__counted_by(extern_const_global_count) arg);
      //
      // void test_local_extern_const_global_count() {
      //   int *__counted_by(extern_const_global_count) local;
      // }
      // ```
      //
      // For the case above the loop below is going over the dependent decls
      // of `extern_const_global_count` when we are running DCPAA on
      // `test_local_extern_const_global_count`. We don't want to add
      // all other uses of the count outside of function (i.e. `arg` in this
      // example) otherwise we will emit an incorrect diagnostic (`local
      // variable arg must be declared right next to its dependent decl`) which
      // doesn't make sense because `arg` isn't in the function being analyzed.
      if (!clang::IsConstOrLateConst(VD)) {
        if (auto *Att = VD->getAttr<DependerDeclsAttr>()) {
          for (const auto *D : Att->dependerDecls()) {
            // XXX: We could've just make it ValueDecl.
            assert(isa<ValueDecl>(D));
            const auto *DepVD = cast<ValueDecl>(D);
            AddDependentDecls(DepVD);
          }
        }
      }

      const auto *DCPTy = VD->getType()->getAs<CountAttributedType>();
      if (!DCPTy && VD->getType()->isPointerType()) {
        DCPTy = VD->getType()->getPointeeType()->getAs<CountAttributedType>();
      }

      if (!DCPTy)
        return;

      for (const auto &DI : DCPTy->dependent_decls()) {
        const auto *D = DI.getDecl();
        assert(isa<ValueDecl>(D));
        const auto *DepVD = cast<ValueDecl>(D);
        AddDependentDecls(DepVD);
      }
    };

    AddDependentDecls(VD);
  }

  static CountDepGroup *Create(const ValueDecl *VD, StringRef Key,
                               StringRef FlexBaseKey,
                               const RecordDecl *FlexBaseDecl) {
    return new CountDepGroup(VD, Key, FlexBaseKey, FlexBaseDecl);
  }

  /// LLVM RTTI Interface
  static bool classof(const DepGroup *DG) {
    return DG->getKind() == DGK_Count;
  }
};

class RangeDepGroup : public DepGroup {
protected:
  void EmitChecksToAST(Sema &SemaRef, ParentMap &PM) override {
    if (!getFirstAssign() || SemaRef.hasUncompilableErrorOccurred())
      return;
    else if (const auto *AE = dyn_cast<BinaryOperator>(getFirstAssign())) {
      assert(BinaryOperator::isAssignmentOp(AE->getOpcode()));
      if (SemaRef.allowBoundsUnsafePointerAssignment(
              AE->getLHS()->getType(), AE->getRHS(), AE->getExprLoc()))
        return;
    }
    PreAssignCheck Builder(SemaRef, PM, getFirstAssign(), DeclToNewValue);
    bool HadOldRangeCheck = false;
    /// If there is only one actual update, we just check the range of that actual assignment.
    if (hasSelfAssign() && AssignedDecls.size() == SelfAssignedDecls.size() + 1) {
      for (const auto &ExprInfo : AssignExprList) {
        // XXX: We should just prevent having multiple assignments to the same decl in the
        // same analysis region (or basic block).
        if (!ExprInfo.IsSelfAssign()) {
          if (SemaRef.allowBoundsUnsafePointerAssignment(
                  ExprInfo.getValueDecl()->getType(),
                  DeclToNewValue[ExprInfo.getValueDecl()].first,
                  ExprInfo.getExpr()->getExprLoc()))
            return;
          HadOldRangeCheck = Builder.buildAndChainOldRangeCheck(ExprInfo.getExpr(),
                                                                DeclToNewValue[ExprInfo.getValueDecl()].first);
          break;
        }
      }
    }
    if (!HadOldRangeCheck) {
      for (auto &ExprInfo : llvm::reverse(AssignExprList)) {
        if (SemaRef.allowBoundsUnsafePointerAssignment(
                ExprInfo.getValueDecl()->getType(),
                DeclToNewValue[ExprInfo.getValueDecl()].first,
                ExprInfo.getExpr()->getExprLoc()))
          return;
        Builder.buildAndChain(ExprInfo.getExpr(),
                              DeclToNewValue[ExprInfo.getValueDecl()].first);
      }
    }
    FinalizePreAssignCheck(SemaRef, PM, Builder.getChainedBoundsCheckExpr());
  }

  bool DiagnoseMissingDependentAssign(Sema &SemaRef,
                                      const ValueDecl *VD) const override {
    unsigned PtrLevel = 0;
    const auto *DRPTy = VD->getType()->getAs<DynamicRangePointerType>();
    if (!DRPTy && VD->getType()->isPointerType()) {
      DRPTy = VD->getType()->getPointeeType()->getAs<DynamicRangePointerType>();
      PtrLevel = 1;
    }
    if (!DRPTy)
      return false;
    assert(DRPTy);
    unsigned FirstAssignIndex = 0;
    bool HasDepAssign = false;
    QualType DependerType;
    const ValueDecl *AssignedDecl = nullptr;
    unsigned AssigneeLevel = 0;
    unsigned VDLevel = 0;
    auto FindPrecedingAssignIndex = [&](const TypeCoupledDeclRefInfo &DI,
                                        const DynamicRangePointerType *DRPTy) {
      if (PtrLevel != (unsigned)DI.isDeref())
        return;
      const auto *D = DI.getDecl();
      assert(isa<ValueDecl>(D));
      const ValueDecl *KeyVD = cast<ValueDecl>(D);
      auto It = AssignedDecls.find(KeyVD);
      if (It != AssignedDecls.end() &&
          (!HasDepAssign || FirstAssignIndex < It->second)) {
        HasDepAssign = true;
        FirstAssignIndex = It->second;
        AssignedDecl = KeyVD;
        AssigneeLevel = DI.isDeref();
        DependerType = DRPTy ? QualType(DRPTy, 0) : cast<ValueDecl>(D)->getType();
        DependerType = getInnerType(DependerType, PtrLevel);
      }
    };
    for (const auto &DI : DRPTy->getEndPtrDecls()) {
      FindPrecedingAssignIndex(DI, DRPTy);
    }
    for (const auto &DI : DRPTy->getStartPtrDecls()) {
      FindPrecedingAssignIndex(DI, nullptr);
    }
    if (HasDepAssign) {
      const auto &ExprInfo = AssignExprList[FirstAssignIndex];
      const Expr *DependentAssign = ExprInfo.getExpr();
      if (!SemaRef.allowBoundsUnsafeAssignment(DependentAssign->getExprLoc()))
        SemaRef.Diag(DependentAssign->getExprLoc(),
                     diag::err_bounds_safety_missing_dependent_var_assignment)
            << getPrefixedName(AssignedDecl, AssigneeLevel)
            << getPrefixedName(VD, VDLevel) << DependerType << 0;
    }
    return HasDepAssign;
  }

public:
  explicit RangeDepGroup(const ValueDecl *VD, StringRef Key)
      : DepGroup(DGK_Range, Key, StringRef()) {
    std::function<void(const ValueDecl*)> AddDependentDecls;
    AddDependentDecls = [this, &AddDependentDecls](const ValueDecl *VD) {
      if (!KeyDecls.insert(VD).second)
        return;

      const auto *DRPTy = VD->getType()->getAs<DynamicRangePointerType>();
      if (!DRPTy && VD->getType()->isPointerType()) {
        DRPTy = VD->getType()->getPointeeType()->getAs<DynamicRangePointerType>();
      }

      if (!DRPTy)
        return;

      for (const auto &DI : DRPTy->endptr_decls()) {
        const auto *D = DI.getDecl();
        assert(isa<ValueDecl>(D));
        const auto *DepVD = cast<ValueDecl>(D);
        AddDependentDecls(DepVD);
      }

      for (const auto &DI : DRPTy->startptr_decls()) {
        const auto *D = DI.getDecl();
        assert(isa<ValueDecl>(D));
        const auto *DepVD = cast<ValueDecl>(D);
        AddDependentDecls(DepVD);
      }
    };

    AddDependentDecls(VD);
  }

  static RangeDepGroup *Create(const ValueDecl *VD, StringRef Key) {
    return new RangeDepGroup(VD, Key);
  }

  /// LLVM RTTI Interface
  static bool classof(const DepGroup *DG) {
    return DG->getKind() == DGK_Range;
  }
};

/// CheckCountAttributedDeclAssignments - This tracks a group of assignments to
/// decl with dynamic count dependency, assuming each Stmt is analyzed backward
/// through "BeginStmt(Stmt)". This reports an error if there is a side effect
/// between assignments within the same group. For example,
/// @code
/// int len;
/// int *__counted_by(len + 1) ptr;
/// ...
/// len = ...;
/// function_with_side_effect();
/// ptr = ...;
/// @endcode
/// "len" and "ptr" are in the same DepGroup and the analyzer detects the side
/// effect in between assignments to these variables in the same group and
/// reports an error.
class CheckCountAttributedDeclAssignments
    : public RecursiveASTVisitor<CheckCountAttributedDeclAssignments> {

  using BaseVisitor = RecursiveASTVisitor<CheckCountAttributedDeclAssignments>;

  Sema &SemaRef;
  AnalysisDeclContext &AC;
  std::unique_ptr<DepGroup> CurrentDepGroup;
  bool InAssignToDepRHS = false;
  unsigned AssigneeLevel = 0;
  QualType AssigneeType;
  bool EarlyLookup = false;
  llvm::SmallPtrSet<Stmt *, 1> DelayedStmts;
  llvm::SmallPtrSet<InitListExpr *, 1> VisitedILEInCompoundLiteralExprs;
  llvm::SmallPtrSet<Expr *, 1> DiagnosedPtrIncDec;

  void Initialize() {
    InAssignToDepRHS = false;
    AssigneeLevel = 0;
    AssigneeType = QualType();
  }

  void PushDelayedStmt(Stmt *S) { DelayedStmts.insert(S); }

  bool CheckAndPopDelayedStmt(Stmt *S) { return DelayedStmts.erase(S); }

  bool MatchDepGroup(const AssignedDeclRefResult &Result) {
    if (!CurrentDepGroup)
      return false;

    if (Result.IsFlexBase) {
      // XXX: I could just calculate the exact base from the beginning?
      std::string FlexBaseKey =
          Result.Key + Result.ThisVD->getName().str() + "->";
      return CurrentDepGroup->matchFlexBaseKey(FlexBaseKey);
    }
    if (!Result.FlexBaseKey.empty())
      return CurrentDepGroup->matchFlexBaseKey(Result.FlexBaseKey);

    return CurrentDepGroup->matchKey(Result.Key, Result.ThisVD);
  }

  /// Assignee is either DeclRef or MemberExpr that is LHS of assignment.
  bool InsertAssignedDecl(const AssignedDeclRefResult &Result,
                          bool IsSelfAssign,
                          bool IsVarDecl = false);
  void RegisterDeclAssignment(Expr *AssignExpr, AssignedDeclRefResult &Result,
                              bool IsSelfAssign);

  void UpdateLastAssign(Expr *E, const AssignedDeclRefResult &Result,
                        bool IsSelfAssign) {
    if (EarlyLookup)
      return;
    assert(CurrentDepGroup && "Empty dependent length");
    CurrentDepGroup->updateLastAssign(E, Result.ThisVD, IsSelfAssign,
                                      Result.IsCountVar, Result.IsCountForFam,
                                      Result.IsFlexBase);
  }

  void SetSideEffectAfter() {
    if (EarlyLookup)
      return;
    if (CurrentDepGroup)
      CurrentDepGroup->setSideEffectAfter();
  }

  void FinalizeDepGroup() {
    if (CurrentDepGroup) {
      CurrentDepGroup->Finalize(SemaRef, getParentMap());
      CurrentDepGroup.reset();
    }
  }

  bool HasDepGroup() const { return CurrentDepGroup != nullptr; }

  bool InCountDepGroup() const { return CurrentDepGroup && isa<CountDepGroup>(CurrentDepGroup); }
  bool InRangeDepGroup() const { return CurrentDepGroup && isa<RangeDepGroup>(CurrentDepGroup); }
  void HandleRHSOfAssignment(BinaryOperator *E);
  void HandleVarInit(VarDecl *VD);
  void ProcessDeclRefInRHSRecursive(Expr *E);

  ParentMap &getParentMap() { return AC.getParentMap(); }

public:
  explicit CheckCountAttributedDeclAssignments(Sema &SemaRef,
                                               AnalysisDeclContext &AC)
      : SemaRef(SemaRef), AC(AC) {}

  ~CheckCountAttributedDeclAssignments() {
    /// Finalize last assign expressions.
    FinalizeDepGroup();
  }

  bool VisitCallExpr(CallExpr *E);
  bool TraverseBinaryOperator(BinaryOperator *E);
  bool TraverseCompoundAssignOperator(CompoundAssignOperator *E) {
    return TraverseBinaryOperator(E);
  }
  bool TraverseUnaryOperator(UnaryOperator *E);
  bool TraverseStmtExpr(StmtExpr *E) { return true; }
  bool TraverseOpaqueValueExpr(OpaqueValueExpr *E) {
    return TraverseStmt(E->getSourceExpr());
  }
  bool TraverseBoundsCheckExpr(BoundsCheckExpr *E) {
    return TraverseStmt(E->getGuardedExpr());
  }
  bool TraverseMaterializeSequenceExpr(MaterializeSequenceExpr *E) {
    return TraverseStmt(E->getWrappedExpr());
  }
  bool TraverseBoundsSafetyPointerPromotionExpr(BoundsSafetyPointerPromotionExpr *E) {
    return TraverseStmt(E->getSubExpr());
  }
  bool VisitVarDecl(VarDecl *VD);
  bool TraverseDeclStmt(DeclStmt *S);
  bool VisitCompoundLiteralExpr(CompoundLiteralExpr *CLE);
  bool TraverseReturnStmt(ReturnStmt *S);

  Expr *HandleInitListExpr(InitListExpr *IL, PreAssignCheck &Builder);
  void BeginStmt(Stmt *S);
};
} // namespace

/// This function generates the flexible struct base.
/// This function finds the flexible struct pointer base that can be updated
/// with a bounded pointer. For `a.c.f->s.b.count = xxx`, as an example,
/// `a.c.f->` is identified as the flexible pointer base. If there is no pointer
/// base, this returns the entire struct base.
/// The access path is emitted to @p OS if provided.
/// The RecordDecl of the struct containing the flexible array member is
/// returned if found.
RecordDecl *DynamicCountPointerAssignmentAnalysis::computeFlexBaseKey(
    Expr *InE, llvm::raw_string_ostream *OS) {
  Expr *E = InE->IgnoreParenCasts();
  RecordDecl *BaseRD = nullptr;
  while (auto ME = dyn_cast<MemberExpr>(E)) {
    // Stripping implicit casts of the base allows us to distinguish between
    // __bidi_indexable vs. __single implicitly promoted to __bidi_indexable.
    QualType BaseTy = ME->getBase()->IgnoreImpCasts()->getType();
    // The closest pointer base becomes the flexible base.
    if (ME->isArrow()) {
      // The base is a pointer, but since we ignore implicit casts we
      // skip over the decay from array to pointer type.
      assert(BaseTy->isPointerType() || BaseTy->isArrayType());
      if (!BaseTy->isSinglePointerType() || BaseTy->isBoundsAttributedType())
        return nullptr;
      if (auto *RD = BaseTy->getPointeeType()->getAsRecordDecl()) {
        if (!RD->hasFlexibleArrayMember())
          return nullptr;
        if (OS)
          computeMemberExprKey(ME, *OS);
        return RD;
      }
      return nullptr;
    }
    BaseRD = BaseTy->getAsRecordDecl();
    E = ME->getBase()->IgnoreParenCasts();
  }

  // There is no pointer base. Emit the whole lvalue base.
  auto *EndME = dyn_cast<MemberExpr>(InE->IgnoreParenCasts());
  if (!EndME || !BaseRD->hasFlexibleArrayMember())
    return nullptr;
  if (OS)
    computeMemberExprKey(EndME, *OS);
  return BaseRD;
}

static DepGroup *CreateDepGroup(const AssignedDeclRefResult &Result) {
  if (Result.IsRangePtrVar)
    return RangeDepGroup::Create(Result.ThisVD, Result.Key);
  else
    return CountDepGroup::Create(Result.ThisVD, Result.Key, Result.FlexBaseKey,
                                 Result.FlexBaseDecl);
}

bool CheckCountAttributedDeclAssignments::InsertAssignedDecl(
    const AssignedDeclRefResult &Result, bool IsSelfAssign, bool IsVarDecl) {
  if (EarlyLookup)
    return false;

  /// If the current group tracks assignments, finalize it.
  if (!MatchDepGroup(Result) ||
      (IsVarDecl && CurrentDepGroup && CurrentDepGroup->getLastAssign())) {
    FinalizeDepGroup();
    CurrentDepGroup.reset(CreateDepGroup(Result));
  }

  AssigneeLevel = Result.Level;
  AssigneeType = Result.Ty;
  if (CurrentDepGroup->hasSideEffectAfter()) {
    SourceLocation Loc = CurrentDepGroup->getLastAssign()
                             ? CurrentDepGroup->getLastAssign()->getExprLoc()
                             : Result.Loc;
    SourceRange Range =
        CurrentDepGroup->getLastAssign()
            ? SourceRange(Result.Loc,
                          CurrentDepGroup->getLastAssign()->getEndLoc())
            : SourceRange();

    SemaRef.Diag(Loc, diag::err_bounds_safety_dependent_vars_assign_side_effect)
        << Range;
    /// Keep tracking the dep group as the error has been reported.
    CurrentDepGroup->clearSideEffectAfter();
  }
  const ValueDecl *VD = Result.ThisVD;
  assert(VD);
  return CurrentDepGroup->insertAssignedDecl(VD, IsSelfAssign);
}

void CheckCountAttributedDeclAssignments::BeginStmt(Stmt *S) {
  Initialize();
  /// Currently, we only detect a group of adjacent simple assignments.
  switch (S->getStmtClass()) {
  case Stmt::DeclRefExprClass:
  case Stmt::BinaryOperatorClass:
  case Stmt::CompoundAssignOperatorClass:
  case Stmt::CallExprClass:
  case Stmt::UnaryExprOrTypeTraitExprClass:
  case Stmt::UnaryOperatorClass:
  case Stmt::DeclStmtClass:
    break;
  default:
    FinalizeDepGroup();
  }
  TraverseStmt(S);
}

namespace {

Expr *findSourceExpr(Expr *E) {
  if (!E)
    return nullptr;
  E = E->IgnoreParens();
  if (auto CE = dyn_cast<ImplicitCastExpr>(E)) {
    if (CE->getCastKind() == CK_BoundsSafetyPointerCast)
      return CE->getSubExpr();
  }
  return E;
}

// Get the mapping from dependent fields and non-param variables to their
// initializers. If a field/variable doesn't have an initializer, create an
// implicit one (zero-init).
Sema::DependentValuesMap getDependentInits(Sema &SemaRef,
                                           const CountAttributedType *DCPT,
                                           InitListExpr *ILE) {
  Sema::DependentValuesMap DependentValues;
  for (auto &DI : DCPT->dependent_decls()) {
    // Fields and non-param variables cannot have dependent inout counts.
    assert(!DI.isDeref());

    ValueDecl *D = DI.getDecl();
    auto *VD = dyn_cast<VarDecl>(D);
    auto *FD = dyn_cast<FieldDecl>(D);
    assert(FD || (VD && !isa<ParmVarDecl>(VD)));

    // We allow the dependent decl to be an __unsafe_late_const, but we don't
    // know its value, thus we don't replace it.
    if (VD && VD->hasAttr<UnsafeLateConstAttr>())
      continue;

    Expr *Init;
    if (VD && VD->hasInit()) {
      Init = VD->getInit();
    } else if (FD && ILE && FD->getFieldIndex() < ILE->getNumInits()) {
      Init = ILE->getInit(FD->getFieldIndex());
    } else {
      // Assume zero-init when there is no initializer.
      Init = new (SemaRef.Context) ImplicitValueInitExpr(D->getType());
    }

    DependentValues[D] = {Init, /*Level=*/0};
  }
  return DependentValues;
}

bool diagnoseDynamicCountVarInit(Sema &SemaRef, SourceLocation Loc,
                                 const Twine &Designator, QualType Ty,
                                 Expr *Init, InitListExpr *ParentInit) {
  const auto *DCPT = Ty->getAs<CountAttributedType>();
  if (DCPT && Ty->isPointerType()) {
    auto DependentValues = getDependentInits(SemaRef, DCPT, ParentInit);

    Expr *RHS;
    if (Init)
      RHS = Init->IgnoreParenImpCasts();
    else
      RHS = new (SemaRef.Context) ImplicitValueInitExpr(Ty);

    return !SemaRef.CheckDynamicCountSizeForAssignment(
        Ty, RHS, AssignmentAction::Initializing, Loc, Designator,
        DependentValues);
  }

  auto GetChildLoc = [&](Expr *ChildInit) {
    // If we have an explicit init, get the location from it.
    if (ChildInit && !isa<ImplicitValueInitExpr>(ChildInit))
      return ChildInit->getBeginLoc();

    // For implicit inits, we point to the '}' in the parent initializer list,
    // for example:
    //   struct foo { int len; int *__counted_by(len) p; };
    //   struct foo f = { 1 };
    //                      ^ diagnostic for 'p' here
    if (const auto *ILE = dyn_cast_or_null<InitListExpr>(Init))
      return ILE->getEndLoc();

    // Otherwise, point at the parent location (e.g., the VarDecl's location).
    return Loc;
  };

  auto GetInitListExpr = [&SemaRef](Expr *Init) -> InitListExpr * {
    if (!Init)
      return nullptr;
    auto *ILE = dyn_cast<InitListExpr>(Init);
    if (ILE)
      return ILE;

    // Try CompoundLiteralExpr assignment.
    // E.g.
    // struct Foo a = (struct Foo) { .field = 0x0 };

    // Checking CompoundLiteralExpr assignment is currently guarded by the
    // new bounds check to avoid potential build failures.
    // TODO: We should **always** check CompoundLiteralExprs (rdar://138982703).
    if (!SemaRef.getLangOpts().hasNewBoundsSafetyCheck(
            LangOptions::BS_CHK_CompoundLiteralInit))
      return nullptr;
    auto *CLE =
        dyn_cast_if_present<CompoundLiteralExpr>(Init->IgnoreImpCasts());
    if (!CLE)
      return nullptr;
    ILE = dyn_cast_if_present<InitListExpr>(CLE->getInitializer());
    return ILE;
  };

  if (const auto *RD = Ty->getAsRecordDecl()) {
    assert(RD->isStruct() || RD->isUnion());

    auto *ILE = GetInitListExpr(Init);

    // If RD is an union and we know which field is being initialized, check
    // only that field and ignore the rest.
    if (RD->isUnion() && ILE) {
      const FieldDecl *FD = ILE->getInitializedFieldInUnion();
      Expr *ChildInit = ILE->getNumInits() > 0 ? ILE->getInit(0) : nullptr;
      return diagnoseDynamicCountVarInit(
          SemaRef, GetChildLoc(ChildInit),
          Designator.concat(llvm::Twine('.').concat(FD->getName())),
          FD->getType(), ChildInit, ILE);
    }

    bool HadError = false;
    for (const auto *FD : RD->fields()) {
      Expr *ChildInit = nullptr;
      if (ILE && FD->getFieldIndex() < ILE->getNumInits())
        ChildInit = ILE->getInit(FD->getFieldIndex());
      HadError |= diagnoseDynamicCountVarInit(
          SemaRef, GetChildLoc(ChildInit),
          Designator.concat(llvm::Twine('.').concat(FD->getName())),
          FD->getType(), ChildInit, ILE);
    }
    return HadError;
  }

  if (const auto *CAT = SemaRef.Context.getAsConstantArrayType(Ty)) {
    assert(CAT->getSize().isNonNegative());
    uint64_t Size = CAT->getSize().getZExtValue();
    auto *ILE = GetInitListExpr(Init);

    // Don't interate over the array elements if the element type is
    // uninteresting for the diagnosis.
    QualType ET = CAT->getElementType();
    if (!ET->isCountAttributedType() && !ET->getAsRecordDecl() &&
        !SemaRef.Context.getAsConstantArrayType(ET)) {
      return false;
    }

    // Avoid complaining multiple times for implicit init, since the errors are
    // the same.
    uint64_t MaxSize = !ILE ? 1 : (ILE->getNumInits() + 1);
    if (MaxSize < Size)
      Size = MaxSize;

    bool HadError = false;
    for (uint64_t I = 0; I < Size; ++I) {
      Expr *ChildInit = nullptr;
      if (ILE && I < ILE->getNumInits())
        ChildInit = ILE->getInit(I);
      HadError |= diagnoseDynamicCountVarInit(
          SemaRef, GetChildLoc(ChildInit),
          Designator.concat(llvm::Twine('[')
                                .concat(llvm::utostr(I))
                                .concat(llvm::Twine(']'))),
          ET, ChildInit, nullptr);
    }
    return HadError;
  }

  return false;
}

} // namespace

bool Sema::DiagnoseDynamicCountVarZeroInit(VarDecl *VD) {
  if (VD->hasExternalStorage())
    return false;

  return diagnoseDynamicCountVarInit(*this, VD->getLocation(), VD->getName(),
                                     VD->getType(), VD->getInit(),
                                     /*ParentInit=*/nullptr);
}

namespace {

bool diagnoseRecordInitsImpl(
    Sema &SemaRef, InitListExpr *IL, bool &NeedPreCheck,
    bool DiagnoseAssignments,
    SmallVectorImpl<Expr *> &InitializersWithSideEffects) {
  bool HadError = false;

  if (SemaRef.Context.getAsArrayType(IL->getType())) {
    for (unsigned i = 0; i < IL->getNumInits(); ++i) {
      if (auto SubIL = dyn_cast<InitListExpr>(IL->getInit(i))) {
        HadError |= diagnoseRecordInitsImpl(SemaRef, SubIL, NeedPreCheck,
                                            DiagnoseAssignments,
                                            InitializersWithSideEffects);
      }
    }
    return HadError;
  }

  auto RD = IL->getType()->getAsRecordDecl();
  if (!RD)
    return true;

  if (RD->isUnion()) {
    auto FD = IL->getInitializedFieldInUnion();
    if (FD && IL->getNumInits()) {
      if (auto SubIL = dyn_cast<InitListExpr>(IL->getInit(0))) {
        HadError |= diagnoseRecordInitsImpl(SemaRef, SubIL, NeedPreCheck,
                                            DiagnoseAssignments,
                                            InitializersWithSideEffects);
      }
    }
    return HadError;
  }

  // We memoize null-pointer evaluation results for each init so we don't
  // repeat the same evaluation.
  enum NullState { NS_Unknown, NS_Null, NS_Nonnull };
  llvm::SmallVector<NullState, 2> InitIsNull(IL->getNumInits(), NS_Unknown);
  unsigned InitIndex = 0;
  for (auto FD : RD->fields()) {
    assert(InitIndex < IL->getNumInits());
    unsigned CurrInitIndex = InitIndex;
    Expr *Init = IL->getInit(InitIndex++);
    if (auto SubIL = dyn_cast<InitListExpr>(Init)) {
      HadError |= diagnoseRecordInitsImpl(SemaRef, SubIL, NeedPreCheck,
                                          DiagnoseAssignments,
                                          InitializersWithSideEffects);
      continue;
    }
    if (SemaRef.allowBoundsUnsafePointerAssignment(FD->getType(), Init,
                                                   Init->getExprLoc()))
      continue;

    auto isNullPtrInit = [&](unsigned Idx, ASTContext &Context) -> bool {
      assert(Idx < IL->getNumInits());
      if (InitIsNull[Idx] != NS_Unknown)
        return InitIsNull[Idx] == NS_Null;
      Expr *ThisInit = IL->getInit(Idx);
      bool IsNull =
          isa<ImplicitValueInitExpr>(ThisInit) ||
          ThisInit->isNullPointerConstant(
              Context, Expr::NPC_ValueDependentIsNotNull) != Expr::NPCK_NotNull;
      InitIsNull[Idx] = IsNull ? NS_Null : NS_Nonnull;
      return IsNull;
    };

    // XXX: rdar://76568300
    if (Init->HasSideEffects(SemaRef.Context)) {
      InitializersWithSideEffects.emplace_back(Init);
      SourceLocation Loc = Init->getExprLoc();
      if (auto *Att = FD->getAttr<DependerDeclsAttr>()) {
        NeedPreCheck = true;
        if (!Att->getIsDeref()) {
          assert(Att->dependerDecls_size() > 0);
          const auto *Depender = *Att->dependerDecls_begin();
          assert(isa<ValueDecl>(Depender));
          const auto *DepVD = cast<ValueDecl>(Depender);
          const auto *DepDCPTy = DepVD->getType()->getAs<CountAttributedType>();
          assert(DepDCPTy);
          SemaRef.Diag(Loc, diag::err_bounds_safety_dynamic_bound_init_side_effect)
          << (DepDCPTy->isCountInBytes() ? 1 : 0);
          HadError = true;
          continue;
        }
      }
      if (const auto *DCPTy = FD->getType()->getAs<CountAttributedType>()) {
        NeedPreCheck = true;
        SemaRef.Diag(Loc, diag::err_bounds_safety_dynamic_bound_init_side_effect)
            << DCPTy->getKind() + 2;
        HadError = true;
        continue;
      }
      if (const auto *DRPTy = FD->getType()->getAs<DynamicRangePointerType>()) {
        NeedPreCheck = true;
        SemaRef.Diag(Loc, diag::err_bounds_safety_dynamic_bound_init_side_effect)
            << (DRPTy->getEndPointer() ? 6 : 7);
        HadError = true;
        continue;
      }
    }

    if (auto *OrigDCPTy = FD->getType()->getAs<CountAttributedType>()) {
      NeedPreCheck = true;
      if (OrigDCPTy->isPointerType() && DiagnoseAssignments) {
        // If requested try to diagnose assignments. This is currently
        // only for CompoundLiteralExpr initializer lists. Variable initializers
        // are handled elsewhere (DepGroup::CheckDynamicCountAssignments).
        auto DependentValues = getDependentInits(SemaRef, OrigDCPTy, IL);
        HadError |= !SemaRef.CheckDynamicCountSizeForAssignment(
            FD->getType(), Init, AssignmentAction::Initializing,
            Init->getExprLoc(), StringRef(), DependentValues);
      }
    } else if (auto *OrigDRPTy =
                   FD->getType()->getAs<DynamicRangePointerType>()) {
      NeedPreCheck = true;
      bool IsImplicitInitExpr = isa<ImplicitValueInitExpr>(Init);
      if (isNullPtrInit(CurrInitIndex, SemaRef.Context)) {
        auto diagnosePartialNullRange = [&](const TypeCoupledDeclRefInfo &DepDeclInfo,
                                            Expr *DepExp) {
          if (DepDeclInfo.isDeref())
            return;
          assert(isa<FieldDecl>(DepDeclInfo.getDecl()));
          const auto *DepField = cast<FieldDecl>(DepDeclInfo.getDecl());
          assert(DepField->getFieldIndex() < IL->getNumInits());
          if (!isNullPtrInit(DepField->getFieldIndex(), SemaRef.Context)) {
            SourceLocation Loc = IsImplicitInitExpr ? IL->getEndLoc() : Init->getExprLoc();
            SemaRef.Diag(Loc, diag::warn_bounds_safety_initlist_range_partial_null)
            << IsImplicitInitExpr << FD << FD->getType() << DepExp;
          }
        };
        if (auto *EndPtr = OrigDRPTy->getEndPointer()) {
          assert(isa<DeclRefExpr>(EndPtr->IgnoreParenCasts())
                 || isa<MemberExpr>(EndPtr->IgnoreParenCasts()));
          for (const auto &EndPtrDecl : OrigDRPTy->getEndPtrDecls()) {
            diagnosePartialNullRange(EndPtrDecl, EndPtr);
          }
        }
        if (auto *StartPtr = OrigDRPTy->getStartPointer()) {
          assert(isa<DeclRefExpr>(StartPtr->IgnoreParenCasts())
                 || isa<MemberExpr>(StartPtr->IgnoreParenCasts()));
          for (const auto &StartPtrDecl : OrigDRPTy->getStartPtrDecls()) {
            diagnosePartialNullRange(StartPtrDecl, StartPtr);
          }
        }
      }
    }
  }
  return HadError;
}

bool diagnoseRecordInits(Sema &SemaRef, InitListExpr *IL, bool &NeedPreCheck,
                         bool DiagnoseAssignments = false) {
  SmallVector<Expr *> FieldInitializersWithSideEffects;
  bool Result =
      diagnoseRecordInitsImpl(SemaRef, IL, NeedPreCheck, DiagnoseAssignments,
                              FieldInitializersWithSideEffects);

  // Avoid
  // * Warning about structs that don't contain externally counted attributes.
  // * Warning when we emitted errors. That just adds extra noise.
  if (!NeedPreCheck || Result)
    return Result;

  // This record needs bounds checks and there were no errors. Warn about any
  // initializers that cause side effects.
  for (auto *FE : FieldInitializersWithSideEffects) {
    SemaRef.Diag(FE->getExprLoc(), diag::warn_bounds_safety_struct_init_side_effect)
        << FE;
  }

  return Result;
}

} // namespace

/// Recursively check Inits of InitListExpr and directly insert new AST nodes for pre-assignment checks and
/// necessary materializations.
///
/// This currently doesn't support inits used in the checks to have side effects because this doesn't guarantees
/// that inits are evaluated in order when dependent fields and the other fields are mixed in an InitListExpr.
/// To support it, we need to make sure all inits including nested ones are materialized in the program order
/// before performing the assignment checks.
/// rdar://76568300
///
/// Similarly to assignment expressions, initialization list of structs with
/// dynamic count and/or dynamic count pointer fields needs run-time integrity
/// checks, e.g., the init list like following.
/// @code
/// struct S { int *__counted_by(len) ptr; int len; };
/// struct S s = {wide_p, l};
/// @endcode
/// This function tracks and records wide pointer expression assigned as an
/// element of init list so that CodeGen can insert necessary runtime integrity
/// checks for assigned dynamic count value, e.g., "len <= bound_of(wide_p)".
Expr *CheckCountAttributedDeclAssignments::HandleInitListExpr(
    InitListExpr *IL, PreAssignCheck &Builder) {
  if (IL->getType()->isRecordType()) {
    auto RD = IL->getType()->getAsRecordDecl();
    if (RD->isUnion()) {
      auto FD = IL->getInitializedFieldInUnion();
      if (FD && IL->getNumInits()) {
        if (auto SubIL = dyn_cast<InitListExpr>(IL->getInit(0))) {
          if (auto *NewSubIL = HandleInitListExpr(SubIL, Builder))
            IL->setInit(0, NewSubIL);
        }
      }
      return nullptr;
    }
    llvm::SmallVector<std::pair<QualType, Expr *>, 2>
        CountPointerCheckPairs;
    unsigned InitIndex = 0;
    llvm::SmallVector<OpaqueValueExpr *, 4> MateredExprs;
    for (auto FD : RD->fields()) {
      assert(InitIndex < IL->getNumInits());
      unsigned CurrInitIndex = InitIndex;
      Expr *Init = IL->getInit(InitIndex++);

      // Note: We don't handle CompoundLiteralExpr here which would contain
      // a sub InitListExpr. Instead we rely on VisitCompoundLiteralExpr being
      // called appropriately.
      if (auto SubIL = dyn_cast<InitListExpr>(Init)) {
        if (auto *NewSubIL = HandleInitListExpr(SubIL, Builder))
          IL->setInit(CurrInitIndex, NewSubIL);
        continue;
      }

      if (FD->hasAttr<DependerDeclsAttr>()) {
        auto InitRes = Builder.getMaterializedValueIfNot(Init, &MateredExprs);
        if (!(Init = InitRes.get()))
          return nullptr;
        Builder.insertDeclToNewValue(FD, Init);
      } else if (FD->getType()->isCountAttributedType()) {
        Expr *PtrExpr = findSourceExpr(Init);
        auto PtrExprRes =
            Builder.getMaterializedValueIfNot(PtrExpr, &MateredExprs);
        if (!(PtrExpr = PtrExprRes.get()))
          return nullptr;

        CountPointerCheckPairs.push_back(
            std::make_pair(FD->getType(), PtrExpr));
      } else if (auto *OrigDRPTy =
                     FD->getType()->getAs<DynamicRangePointerType>()) {
        Expr *PtrExpr = findSourceExpr(Init);
        auto PtrExprRes =
            Builder.getMaterializedValueIfNot(PtrExpr, &MateredExprs);
        if (!(PtrExpr = PtrExprRes.get()))
          return nullptr;
        Builder.insertDeclToNewValue(FD, Init);
        // The pointer that has an associated end pointer is responsible to do the bounds checks:
        //  lb(new_ptr) <= start <= new_ptr  <= end <= ub(new_ptr)
        // Thus, we check if the pointer has an end pointer. This is consistent with how we
        // handle assignments (and this is a routine we handle init lists).
        // XXX: This is based on the assumption that only '__ended_by' is exposed to the users
        // and '__started_by' is an implicit attribute the compiler adds based on the corresponding
        // '__ended_by'. We may revisit this if we decide to expose '__started_by' to users.
        if (OrigDRPTy->getEndPointer()) {
          CountPointerCheckPairs.push_back(std::make_pair(FD->getType(), PtrExpr));
        }
      }
    }
    if (!CountPointerCheckPairs.empty()) {
      ExprResult Result = Builder.build(CountPointerCheckPairs, IL, MateredExprs);
      if (!Result.isInvalid())
        return Result.get();
    }
    return nullptr;
  }

  if (SemaRef.Context.getAsArrayType(IL->getType())) {
    for (unsigned i = 0; i < IL->getNumInits(); ++i) {
      if (auto SubIL = dyn_cast<InitListExpr>(IL->getInit(i))) {
        if (auto *NewSubIL = HandleInitListExpr(SubIL, Builder))
          IL->setInit(i, NewSubIL);
      }
    }
  }
  return nullptr;
}

bool CheckCountAttributedDeclAssignments::TraverseDeclStmt(DeclStmt *S) {
  for (auto *I : S->decls()) {
    auto Var = dyn_cast<VarDecl>(I);
    if (!Var)
      return TraverseDecl(I);

    AssignedDeclRefResult Result;
    analyzeVarDecl(SemaRef, Var, Result);
    if (Result.IsFlexBase && Var->getInit()) {
      RegisterDeclAssignment(nullptr, Result, /* IsSelfAssign */ false);
      if (CurrentDepGroup->skipFlexCheck())
        return true;

      Expr *Init = Var->getInit();
      auto *CE = dyn_cast<CastExpr>(Init);
      if (CE && CE->getCastKind() == CK_BoundsSafetyPointerCast &&
          !Init->getType()->isPointerTypeWithBounds()) {
        Init = CE->getSubExpr();
      }
      if (Init->IgnoreParenCasts()->isNullPointerConstant(
              SemaRef.Context, Expr::NPC_NeverValueDependent) !=
          Expr::NPCK_NotNull) {
        CurrentDepGroup->setFlexBaseNull();
        CurrentDepGroup->setSkipFlexCheck();
        return true;
      }
      // -fbounds-safety doesn't check single to single casts. Likewise,
      // we skip checks for single to single with flexible array member.
      // The exception is when both the source and the destination are
      // pointers to struct with flexible array member, because a single
      // pointer to struct with flexible array member promotes to
      // '__bidi_indexable' with the specified count value.
      if (Init->getType()->isSinglePointerType()) {
        assert(!Init->getType()->getAs<BoundsAttributedType>() &&
               !Init->getType()->getAs<ValueTerminatedType>());
        CurrentDepGroup->setSkipFlexCheck();
        return true;
      }
      auto *OVE = OpaqueValueExpr::Wrap(SemaRef.Context, Init);
      CurrentDepGroup->insertMateredExpr(OVE);
      ReplaceSubExpr(getParentMap(), OVE->getSourceExpr(), OVE);
      CurrentDepGroup->insertDeclToNewValue(Var, Var->getInit(), Result.Level);
    } else if (Result.IsTrackedVar) {
      InsertAssignedDecl(Result, /*IsSelfAssign*/false, /*VarDecl*/true);
      if (Var->getType()->isBoundsAttributedType())
        HandleVarInit(Var);
      else
        TraverseStmt(Var->getInit());
    } else {
      if (CurrentDepGroup && !CurrentDepGroup->isDeclInitBasedGroup())
        FinalizeDepGroup();
      TraverseDecl(I);
    }
  }
  return true;
}

bool CheckCountAttributedDeclAssignments::VisitVarDecl(VarDecl *VD) {
  auto InitExpr = VD->getInit();
  if (auto IL = dyn_cast_or_null<InitListExpr>(InitExpr)) {
    // If VD is not local, the check should have already been done at earlier
    // semantic analysis.
    bool NeedPreCheck = false;
    if (!diagnoseRecordInits(SemaRef, IL, NeedPreCheck)
        && VD->hasLocalStorage() && NeedPreCheck) {
      PreAssignCheck::DeclToNewValueTy DeclToNewValue;
      PreAssignCheck Builder(SemaRef, getParentMap(), IL, DeclToNewValue);
      if (auto *NewIL = HandleInitListExpr(IL, Builder)) {
        if (VD->getInit() == InitExpr) {
          VD->setInit(NewIL);
        } else {
          bool Mod = ReplaceSubExpr(getParentMap(), InitExpr, NewIL);
          (void)Mod;
          assert(Mod);
        }
      }
    }
    SetSideEffectAfter();
  }
  return true;
}

bool CheckCountAttributedDeclAssignments::VisitCompoundLiteralExpr(
    CompoundLiteralExpr *CLE) {
  // Note this check prevents us from trying to transform already transformed
  // CompoundLiteralExpr because the initializer list will be a
  // `BoundsCheckExpr` after its transformed.
  auto *ILE = llvm::dyn_cast_if_present<InitListExpr>(CLE->getInitializer());
  if (!ILE)
    return true;

  // Avoid emitting duplicate diagnostics. There are at least two reasons why
  // this can happen:
  //
  // 1. `diagnoseRecordInits` found an error so adding bounds checks was skipped
  //    and then this CompoundLiteralExpr is visited again.
  //
  // 2. New bounds checks are disabled. When this happens the ILE is not
  //    replaced with a BoundsCheckExpr which means this method won't return
  //    early if this CompoundLiteralExpr is visited again.
  //
  //
  // The CFG form of the function being analyzed may contain the same ILE
  // multiple times in different CFG statements which means the same
  // CompoundLiteralExpr can be visited multiple times.
  //
  if (VisitedILEInCompoundLiteralExprs.contains(ILE)) {
    return true;
  }
  VisitedILEInCompoundLiteralExprs.insert(ILE);

  // Skip emitting the bounds check if its not enabled
  // rdar://110871666
  if (!SemaRef.getLangOpts().hasNewBoundsSafetyCheck(
          clang::LangOptionsBase::BS_CHK_CompoundLiteralInit))
    return true;

  // TODO: This diagnostic should always be enabled (i.e. should actually be
  // called before we check for BS_CHK_CompoundLiteralInit being enabled)
  // (rdar://138982703). However the diagnostics are currently guarded to avoid
  // a build failure (rdar://137774167).
  bool NeedPreCheck = false;
  if (diagnoseRecordInits(SemaRef, ILE, NeedPreCheck,
                          /*DiagnoseAssignments=*/true) ||
      !NeedPreCheck)
    return true;

  PreAssignCheck::DeclToNewValueTy EmptyMap;
  PreAssignCheck Builder(SemaRef, getParentMap(), ILE, EmptyMap);
  auto *NewILE = HandleInitListExpr(ILE, Builder);
  if (!NewILE)
    // Note in this case the ILE might still have changed but rather than the
    // whole ILE being replaced one of the assignments might have changed. In
    // this case the ILE will have been modified in-place so there is nothing
    // left to do.
    return true;

  // The ILE needs replacing
  assert(CLE->getInitializer() == ILE &&
         "CLE initializer unexpectedly changed");
  CLE->setInitializer(NewILE);
  getParentMap().setParent(NewILE, CLE);
  return true;
}

bool CheckCountAttributedDeclAssignments::VisitCallExpr(CallExpr *E) {
  if (CheckAndPopDelayedStmt(E))
    return true;

  if (InAssignToDepRHS)
    PushDelayedStmt(E);
  else
    FinalizeDepGroup();

  BaseVisitor::VisitCallExpr(E);
  if (E->HasSideEffects(SemaRef.getASTContext(),
                        /*IncludePossibleEffects*/ true)) {
    SetSideEffectAfter();
  }
  return true;
}

void CheckCountAttributedDeclAssignments::RegisterDeclAssignment(
    Expr *AssignExpr, AssignedDeclRefResult &Result, bool IsSelfAssign) {
  assert(Result.ThisVD);
  if (!InsertAssignedDecl(Result, IsSelfAssign)) {
    const Expr *PrevAssignExpr = CurrentDepGroup->getAssignExprForDecl(Result.ThisVD);
    assert(PrevAssignExpr);
    const unsigned DiagIndex = Result.IsRangePtrVar                       ? 2
                               : (Result.IsOutCount || Result.IsCountVar) ? 0
                                                                          : 1;
    SemaRef.Diag(PrevAssignExpr->getExprLoc(),
                 diag::err_bounds_safety_multiple_assignments_to_dynamic_bound_decl)
        << DiagIndex << Result.ThisVD;
    SemaRef.Diag(AssignExpr->getExprLoc(), diag::note_bounds_safety_previous_assignment);
  }
  UpdateLastAssign(AssignExpr, Result, IsSelfAssign);
}

namespace {

Expr *materializeLValue(ASTContext &Ctx, Expr *LV,
                        SmallVectorImpl<OpaqueValueExpr *> &OVEs) {
  Expr *Unwrapped = LV->IgnoreParenImpCasts();
  if (isa<OpaqueValueExpr>(Unwrapped))
    return LV;

  if (auto *ME = dyn_cast<MemberExpr>(Unwrapped)) {
    if (!isa<OpaqueValueExpr>(ME->getBase())) {
      OVEs.push_back(OpaqueValueExpr::Wrap(Ctx, ME->getBase()));
      ME->setBase(OVEs.back());
    }
  }
  OVEs.push_back(OpaqueValueExpr::Wrap(Ctx, LV));
  return OVEs.back();
}

// A non-inout buf parameter having at least one dependent inout count parameter
// must be immutable.
void checkImplicitlyReadOnlyBuf(Sema &SemaRef, const Expr *E,
                                const AssignedDeclRefResult &Result,
                                bool IsSelfAssign = false) {
  const ValueDecl *VD = Result.ThisVD;
  if (IsSelfAssign || (!Result.IsCountPtrVar && !Result.IsRangePtrVar) ||
      !isa<ParmVarDecl>(VD) || isValueDeclOutBuf(VD)) {
    return;
  }
  if (Result.IsCountPtrVar) {
    const auto *DCPT = Result.Ty->getAs<CountAttributedType>();
    assert(DCPT);
    bool HasOutCount = std::any_of(
        DCPT->dependent_decl_begin(), DCPT->dependent_decl_end(),
        [](const auto &DI) { return isValueDeclOutCount(DI.getDecl()); });
    if (HasOutCount) {
      SemaRef.Diag(E->getBeginLoc(),
                   diag::err_bounds_safety_read_only_dynamic_count_pointer)
          << VD->getName() << DCPT->getKind();
    }
    return;
  }
  if (Result.IsRangePtrVar) {
    const auto *DRPT = Result.Ty->getAs<DynamicRangePointerType>();
    assert(DRPT);
    auto IsDIDeref = [](const TypeCoupledDeclRefInfo &DI) {
      return DI.isDeref();
    };
    int DiagNo = -1;
    if (std::any_of(DRPT->endptr_decl_begin(), DRPT->endptr_decl_end(),
                    IsDIDeref)) {
      DiagNo = 0;
    } else if (std::any_of(DRPT->startptr_decl_begin(),
                           DRPT->startptr_decl_end(), IsDIDeref)) {
      DiagNo = 1;
    }

    if (DiagNo >= 0) {
      SemaRef.Diag(E->getBeginLoc(),
                   diag::err_bounds_safety_read_only_dynamic_range_pointer)
          << VD->getName() << DiagNo;
    }
    return;
  }
}

void checkImplicitlyReadOnlyDependentParamOfReturnType(
    Sema &S, const Expr *E, const AssignedDeclRefResult &Result) {
  bool IsDeref = Result.Level >= 1;
  if (!Result.IsDependentParamOfReturnType || IsDeref)
    return;

  // This should trigger an error, don't emit another one.
  if (Result.IsOutCount)
    return;

  // TODO: This diagnostic check should always be performed (rdar://138982703).
  // The check is currently guarded to avoid potentially breaking the build.
  if (!S.getLangOpts().hasNewBoundsSafetyCheck(LangOptions::BS_CHK_ReturnSize))
    return;

  const ValueDecl *VD = Result.ThisVD;
  const FunctionDecl *FD = cast<FunctionDecl>(VD->getDeclContext());
  const auto *RetCATy = FD->getReturnType()->getAs<CountAttributedType>();
  unsigned Kind = RetCATy ? RetCATy->getKind() : BoundsAttributedType::EndedBy;
  S.Diag(E->getBeginLoc(),
         diag::err_bounds_safety_read_only_dependent_param_in_return)
      << VD->getName() << Kind << FD->getName() << FD->getReturnType();
}

} // namespace

bool CheckCountAttributedDeclAssignments::TraverseUnaryOperator(
    UnaryOperator *E) {
  if (!E->isIncrementDecrementOp())
    return BaseVisitor::TraverseUnaryOperator(E);

  AssignedDeclRefResult LHSResult;
  Expr *SubExpr = E->getSubExpr();
  analyzeAssignedDeclRef(SemaRef, SubExpr, LHSResult);

  checkImplicitlyReadOnlyBuf(SemaRef, E, LHSResult);
  checkImplicitlyReadOnlyDependentParamOfReturnType(SemaRef, E, LHSResult);

  // This check should be done in `checkArithmeticUnaryOpBoundsSafetyPointer`,
  // however if the check is performed there then there will be false positives
  // for pointer inc/dec in the lhs of assignments which aren't actually bounds
  // checked due to rdar://98749526. So we do the check here so that we emit
  // the diagnostic for the cases where bounds checks **are** inserted.
  // TODO: Remove this (rdar://135833598).
  if (!SemaRef.getLangOpts().hasNewBoundsSafetyCheck(
          LangOptions::BS_CHK_IndirectCountUpdate)) {
    if (const auto *CATTy = SubExpr->getType()->getAs<CountAttributedType>()) {
      if (!DiagnosedPtrIncDec.contains(E) &&
          !SemaRef
               .BoundsSafetyCheckCountAttributedTypeHasConstantCountForAssignmentOp(
                   CATTy, SubExpr, E->isIncrementOp())) {
        DiagnosedPtrIncDec.insert(E);
      }
    }
  }

  if (LHSResult.IsTrackedVar) {
    assert(!SubExpr->HasSideEffects(SemaRef.Context));
    RegisterDeclAssignment(E, LHSResult, /*SelfAssign*/false);
    if (CurrentDepGroup->skipFlexCheck())
      return true;

    SmallVector<OpaqueValueExpr *, 2> OVEs;
    auto *LHS = SubExpr;
    if (!isa<OpaqueValueExpr>(SubExpr)) {
      LHS = materializeLValue(SemaRef.Context, LHS, OVEs);
      E->setSubExpr(LHS);
    }

    SourceLocation Loc = E->getBeginLoc();
    QualType Ty = SubExpr->getType();
    llvm::APInt OneVal(/*bitwidth*/SemaRef.Context.getTypeSize(Ty), /*val*/1);
    QualType IncDecTy = Ty->isIntegralOrEnumerationType() ? Ty :
        SemaRef.Context.getIntTypeForBitwidth(OneVal.getBitWidth(), /*IsSigned*/false);
    Expr* IncDecValue = IntegerLiteral::Create(SemaRef.Context, OneVal, IncDecTy,
                                               Loc);
    ExprResult NewBinOp = SemaRef.CreateBuiltinBinOp(Loc,
                                                     (E->isIncrementOp() ? BO_Add : BO_Sub),
                                                     LHS,
                                                     IncDecValue);
    if (NewBinOp.isInvalid())
      return true;
    OVEs.push_back(OpaqueValueExpr::Wrap(SemaRef.Context, NewBinOp.get()));
    CurrentDepGroup->insertDeclToNewValue(LHSResult.ThisVD, OVEs.back(),
                                          LHSResult.Level);
    CurrentDepGroup->insertMateredExprsReverse(OVEs);
    auto Res = TraverseStmt(SubExpr);
    if (LHSResult.IsFlexBase)
      SetSideEffectAfter();
    return Res;
  }
  return BaseVisitor::TraverseUnaryOperator(E);
}

bool CheckCountAttributedDeclAssignments::TraverseBinaryOperator(BinaryOperator *E) {
  if (E->isAssignmentOp()) {
    AssignedDeclRefResult LHSResult;
    analyzeAssignedDeclRef(SemaRef, E->getLHS(), LHSResult);

    // This check should be done in `checkArithmeticBinOpBoundsSafetyPointer`,
    // however if the check is performed there then there will be false
    // positives for pointer inc/dec in the lhs of assignments which aren't
    // actually bounds checked due to rdar://98749526. So we do the check here
    // so that we emit the diagnostic for the cases where bounds checks **are**
    // inserted.
    // TODO: Remove this (rdar://135833598).
    if (!SemaRef.getLangOpts().hasNewBoundsSafetyCheck(
            LangOptions::BS_CHK_IndirectCountUpdate)) {
      if (const auto *CATTy =
              E->getLHS()->getType()->getAs<CountAttributedType>()) {
        switch (E->getOpcode()) {
        case BO_AddAssign: // +=
        case BO_SubAssign: // -=
          if (!DiagnosedPtrIncDec.contains(E) &&
              !SemaRef
                   .BoundsSafetyCheckCountAttributedTypeHasConstantCountForAssignmentOp(
                       CATTy, E->getRHS(), E->getOpcode())) {
            DiagnosedPtrIncDec.insert(E);
          }

          break;
        default:
          // Don't try to emit the diagnostic for other operators
          break;
        }
      }
    }

    if (LHSResult.IsInnerStruct) {
      auto *InnerStructDecl = cast<FieldDecl>(LHSResult.ThisVD);
      auto *Att = InnerStructDecl->getAttr<DependerDeclsAttr>();
      assert(Att);
      auto *RDecl =
          InnerStructDecl->getType()->getAs<RecordType>()->getAsRecordDecl();
      assert(RDecl);
      // Multiple structs with FAMs can refer to fields in the same struct,
      // so the number can be >1
      assert(Att->dependerDecls_size() > 0);
      for (auto DepDecl : Att->dependerDecls()) {
        if (!LHSResult.FlexBaseDecl->isParentStructOf(DepDecl))
          continue;
        ValueDecl *FAMDecl = cast<ValueDecl>(DepDecl);
        auto *DCTy = FAMDecl->getType()->getAs<CountAttributedType>();
        assert(DCTy);
        for (auto Dep : DCTy->dependent_decls()) {
          auto FieldD = dyn_cast<FieldDecl>(Dep.getDecl());
          if (!FieldD || !RDecl->isParentStructOf(FieldD))
            continue;
          SemaRef.Diag(E->getOperatorLoc(),
                       diag::err_bounds_safety_dependent_struct_assignment)
              << E->getLHS() << FieldD << FAMDecl;
          SemaRef.Diag(DCTy->getCountExpr()->getBeginLoc(),
                       diag::note_bounds_safety_count_param_loc)
              << DCTy->getCountExpr()->getSourceRange();
          return true;
        }
        llvm_unreachable(
            "FAM refers to struct without referring to one of it's fields");
      }
    }

    if (LHSResult.IsOutBuf)
      SemaRef.Diag(E->getBeginLoc(),
                    diag::err_bounds_safety_dependent_out_count_buf_assign);

    if (LHSResult.IsOutCount)
      SemaRef.Diag(E->getBeginLoc(),
                    diag::err_bounds_safety_dependent_out_count_assign);

    AssignedDeclRefResult RHSResult;
    analyzeAssignedDeclRef(SemaRef, E->getRHS(), RHSResult);
    // Allow self assignment
    bool IsSelfAssign = LHSResult.IsTrackedVar &&
                        (LHSResult.ThisVD == RHSResult.ThisVD) &&
                        (LHSResult.Key == RHSResult.Key) &&
                        (LHSResult.Level == RHSResult.Level);

    checkImplicitlyReadOnlyBuf(SemaRef, E, LHSResult, IsSelfAssign);
    checkImplicitlyReadOnlyDependentParamOfReturnType(SemaRef, E, LHSResult);

    if (LHSResult.IsTrackedVar)
      RegisterDeclAssignment(E, LHSResult, IsSelfAssign);

    SaveAndRestore<bool> InAssignToDepRHSLocal(InAssignToDepRHS,
                                               LHSResult.IsTrackedVar);

    // We are only interested in the immediate wide pointer to dynamic
    // bound pointer casts, rather than any casts happening during
    // recursive traversal of subexpressions.
    Expr *OrigRHS = E->getRHS();
    if (InAssignToDepRHS && !CurrentDepGroup->skipFlexCheck()) {

      // SelfAssign doesn't need additional recursive decl ref check, but
      // it still needs materialization in HandleRHSOfAssignment.
      if (!IsSelfAssign)
        ProcessDeclRefInRHSRecursive(OrigRHS);

      Expr *NewValue = nullptr;
      if (E->isCompoundAssignmentOp()) {
        auto *LHS = E->getLHS();
        auto *RHS = E->getRHS();
        SmallVector<OpaqueValueExpr *, 3> OVEs;
        // Evaluate RHS first.
        if (!isa<OpaqueValueExpr>(RHS)) {
          RHS = materializeLValue(SemaRef.Context, RHS, OVEs);
          E->setRHS(RHS);
        }
        if (!isa<OpaqueValueExpr>(LHS)) {
          LHS = materializeLValue(SemaRef.Context, LHS, OVEs);
          E->setLHS(LHS);
        }
        BinaryOperator::Opcode AdjustedBinOp;
        switch (E->getOpcode()) {
        case BO_MulAssign: AdjustedBinOp = BO_Mul; break;
        case BO_DivAssign: AdjustedBinOp = BO_Div; break;
        case BO_RemAssign: AdjustedBinOp = BO_Rem; break;
        case BO_AddAssign: AdjustedBinOp = BO_Add; break;
        case BO_SubAssign: AdjustedBinOp = BO_Sub; break;
        case BO_ShlAssign: AdjustedBinOp = BO_Shl; break;
        case BO_ShrAssign: AdjustedBinOp = BO_Shr; break;
        case BO_AndAssign: AdjustedBinOp = BO_And; break;
        case BO_XorAssign: AdjustedBinOp = BO_Xor; break;
        case BO_OrAssign: AdjustedBinOp = BO_Or; break;
        default: llvm_unreachable("unknown compound operator!");
        }
        ExprResult NewBinOp = SemaRef.CreateBuiltinBinOp(
            E->getExprLoc(), AdjustedBinOp, LHS, RHS);
        if (NewBinOp.isInvalid())
          return true;
        NewValue = NewBinOp.get();
        assert(!isa<OpaqueValueExpr>(NewValue));
        OVEs.push_back(OpaqueValueExpr::Wrap(SemaRef.Context, NewValue));
        NewValue = OVEs.back();
        // Inserting OVE in a reversed order to match that the entire analysis
        // is reversed.
        CurrentDepGroup->insertMateredExprsReverse(OVEs);
      } else if (LHSResult.IsFlexBase) {
        Expr *RHS = OrigRHS;
        CastExpr *E = dyn_cast<CastExpr>(RHS);
        if (E && E->getCastKind() == CK_BoundsSafetyPointerCast &&
            !OrigRHS->getType()->isPointerTypeWithBounds()) {
          // This is a best effort to get an underlying wide pointer, but it may
          // not be a wide pointer if there was no wide pointer in the first
          // place and/or it is a null pointer.
          RHS = E->getSubExpr();
        }
        if (RHS->IgnoreParenCasts()->isNullPointerConstant(
                SemaRef.Context, Expr::NPC_NeverValueDependent) !=
            Expr::NPCK_NotNull) {
          CurrentDepGroup->setFlexBaseNull();
          CurrentDepGroup->setSkipFlexCheck();
        } else if (RHS->getType()->isSinglePointerType()) {
          // -fbounds-safety doesn't check single to single casts. Likewise,
          // we skip checks for single to single with flexible array member.
          // The exception is when both the source and the destination are
          // pointers to struct with flexible array member, because a single
          // pointer to struct with flexible array member promotes to
          // '__bidi_indexable' with the specified count value.
          assert(!RHS->getType()->getAs<BoundsAttributedType>() &&
                 !RHS->getType()->getAs<ValueTerminatedType>());
          CurrentDepGroup->setSkipFlexCheck();
        }
      }

      if (!CurrentDepGroup->skipFlexCheck()) {
        HandleRHSOfAssignment(E);
        if (!NewValue) {
          assert(E->isAssignmentOp());
          NewValue = E->getRHS();
        }
        CurrentDepGroup->insertDeclToNewValue(LHSResult.ThisVD, NewValue,
                                              LHSResult.Level);
      }
    }
    // We still traverse RHS to detect possible side effects.
    if (!IsSelfAssign)
      TraverseStmt(OrigRHS);
    if (!LHSResult.IsTrackedVar &&
        SemaRef.getLangOpts().hasNewBoundsSafetyCheck(
            LangOptions::BS_CHK_IndirectCountUpdate))
      TraverseStmt(E->getLHS());

    if (LHSResult.IsFlexBase)
      SetSideEffectAfter();

    return true;
  } else if (E->isCommaOp()) {
    // Skip traversing the operands of comma operators here because the operands
    // will be visited as seperate statements. Without skipping, the same
    // expressions were traversed twice and this produced unstructured OVEs and
    // led to an assertion failure in Clang CodeGen. For example, the block for
    // `p++, len--` should look like this. The code here skips (3) and the
    // traversals are done for (1) and (2). [B1]
    //  1: p++
    //  2: len--
    //  3: [B1.1] , [B1.2] <- comma operator
    return false;
  }
  return BaseVisitor::TraverseBinaryOperator(E);
}

void CheckCountAttributedDeclAssignments::ProcessDeclRefInRHSRecursive(Expr *E) {
  assert(CurrentDepGroup != nullptr);
  struct DeclRefVisitor : public RecursiveASTVisitor<DeclRefVisitor> {
    Sema &SemaRef;
    DepGroup &DepGroupRef;

    DeclRefVisitor(Sema &SemaRef, DepGroup &DepGroupRef)
        : SemaRef(SemaRef), DepGroupRef(DepGroupRef) {}

    bool VisitDeclRefExpr(DeclRefExpr *E) {
      addReferencedDecl(E);
      return true;
    }
    bool VisitMemberExpr(MemberExpr *E) {
      addReferencedDecl(E);
      return true;
    }

    bool TraverseOpaqueValueExpr(OpaqueValueExpr *E) {
      return TraverseStmt(E->getSourceExpr());
    }

    bool TraverseBoundsCheckExpr(BoundsCheckExpr *E) {
      return TraverseStmt(E->getGuardedExpr());
    }

    bool TraverseMaterializeSequenceExpr(MaterializeSequenceExpr *E) {
      return TraverseStmt(E->getWrappedExpr());
    }

    bool TraverseBoundsSafetyPointerPromotionExpr(BoundsSafetyPointerPromotionExpr *E) {
      return TraverseStmt(E->getSubExpr());
    }

    void addReferencedDecl(Expr *E) {
      AssignedDeclRefResult Result;
      analyzeAssignedDeclRef(SemaRef, E, Result);
      if (Result.IsTrackedVar &&
          DepGroupRef.matchKey(Result.Key, Result.ThisVD)) {
        DepGroupRef.addReferencedDecl(Result.ThisVD, Result.Level);
      }
    }
  } DeclRefVisitorInst{SemaRef, *CurrentDepGroup};
  DeclRefVisitorInst.TraverseStmt(E);
}

void CheckCountAttributedDeclAssignments::HandleRHSOfAssignment(BinaryOperator *AssignOp) {
  assert(InAssignToDepRHS);
  ParentMap &PM = getParentMap();
  Expr *RHS = AssignOp->getRHS()->IgnoreParens();
  CastExpr *E = dyn_cast<CastExpr>(RHS);
  if (!E || E->getCastKind() != CK_BoundsSafetyPointerCast) {
    if (!isa<OpaqueValueExpr>(AssignOp->getRHS())) {
      auto *OVE = OpaqueValueExpr::Wrap(SemaRef.Context, AssignOp->getRHS());
      PM.setParent(OVE->getSourceExpr(), OVE);
      CurrentDepGroup->insertMateredExpr(OVE);
      AssignOp->setRHS(OVE);
      PM.setParent(OVE, AssignOp);
    }
    return;
  }

  if (!isa<OpaqueValueExpr>(E->getSubExpr())) {
    auto *OVE = OpaqueValueExpr::Wrap(SemaRef.Context, E->getSubExpr());
    PM.setParent(OVE->getSourceExpr(), OVE);
    CurrentDepGroup->insertMateredExpr(OVE);
    E->setSubExpr(OVE);
    PM.setParent(OVE, E);
  }
  return;
}

void CheckCountAttributedDeclAssignments::HandleVarInit(VarDecl *VD) {
  if (!VD->getInit() || !VD->hasLocalStorage())
    return;
  Expr *Init = VD->getInit()->IgnoreParens();

  // For variable initialization, we can just emit a check immediately because
  // the length variable has necessarily been initialized before (else it
  // couldn't have been referenced in the dynamic bound pointer type).
  // XXX: this will not work for DynamicRangePointerType.
  PreAssignCheck::DeclToNewValueTy EmptyMap;
  PreAssignCheck Builder(SemaRef, getParentMap(), VD->getInit(), EmptyMap);

  CastExpr *E = dyn_cast<CastExpr>(Init);
  if (E && E->getType()->isBoundsAttributedType()) {
    Init = E->getSubExpr();
  }
  ExprResult Result = Builder.build(VD->getType(), Init);
  if (Result.isInvalid())
    return;

  auto Res = Result.get();
  if (Res->getType()->isPointerTypeWithBounds()) {
    assert(VD->getType()->isBoundsAttributedType());
    Res = SemaRef
              .ImpCastExprToType(Res, VD->getType(),
                                 CastKind::CK_BoundsSafetyPointerCast)
              .get();
  }
  VD->setInit(Res);
}

// Emit a bounds check if the function returns bounds-attributed pointer type.
// The bounds check will load all dependent decls and verify if the dynamic
// bound pointer has enough bytes.
bool CheckCountAttributedDeclAssignments::TraverseReturnStmt(ReturnStmt *S) {
  const auto *FD = cast<FunctionDecl>(AC.getDecl());

  QualType RetTy = FD->getReturnType();
  const auto *RetBATy = RetTy->getAs<BoundsAttributedType>();
  if (!RetBATy)
    return BaseVisitor::TraverseReturnStmt(S);

  auto *RetVal = S->getRetValue();
  if (!RetVal || RetVal->containsErrors()) {
    // This should have been diagnosed earlier.
    return true;
  }

  // Return size checks are hidden behind a flag.
  if (!SemaRef.getLangOpts().hasNewBoundsSafetyCheck(
          LangOptions::BS_CHK_ReturnSize)) {
    return BaseVisitor::TraverseReturnStmt(S);
  }

  // Remove the cast to return type (bounds attribute type) that can be added by
  // Sema. This results in a more readable AST because
  // the pointer used in bounds checks doesn't have the bounds attribute. E.g.
  // having the `__counted_by` attribute on a pointer used in bounds checks
  // before it's been checked is confusing.
  auto *ICE = dyn_cast<ImplicitCastExpr>(RetVal);
  if (ICE && RetVal->getType()->isBoundsAttributedType())
    RetVal = ICE->getSubExpr();

  SmallVector<OpaqueValueExpr *, 4> OVEs;

  // Wrap the return value into OVE to evaluate the value only once.
  auto *RetOVE = OpaqueValueExpr::Wrap(SemaRef.Context, RetVal);
  OVEs.push_back(RetOVE);

  // Add implicit cast to the guarded expression so that it matches the return
  // type. This makes the AST more readable but isn't necessarily required by
  // codegen.
  CastKind Kind = CK_BoundsSafetyPointerCast;
  if (!RetVal->getType()->isPointerType()) {
    assert(RetVal->isNullPointerConstant(SemaRef.getASTContext(),
                                         Expr::NPC_NeverValueDependent));
    Kind = CK_NullToPointer;
  }
  ExprResult GuardedRes = SemaRef.ImpCastExprToType(RetOVE, RetTy, Kind);
  if (GuardedRes.isInvalid())
    return true;

  PreAssignCheck::DeclToNewValueTy EmptyMap;
  PreAssignCheck Builder(SemaRef, getParentMap(), GuardedRes.get(), EmptyMap);

  // Wrap all dependent decls into OVEs, since the bounds check for __ended_by()
  // uses the end pointer value multiple times.
  for (const auto &DI : RetBATy->dependent_decls()) {
    ValueDecl *VD = DI.getDecl();

    // The decl should be either ParmVarDecl or VarDecl that has
    // __unsafe_late_const attribute.
    assert(isa<VarDecl>(VD));

    // Load the decl.
    DeclarationNameInfo NameInfo(VD->getDeclName(), RetVal->getBeginLoc());
    auto *DRE = DeclRefExpr::Create(SemaRef.Context, NestedNameSpecifierLoc(),
                                    /*TemplateKWLoc=*/SourceLocation(), VD,
                                    /*RefersToCapturedVariable=*/false,
                                    NameInfo, VD->getType(), VK_LValue);
    auto Res = SemaRef.DefaultLvalueConversion(DRE);
    if (Res.isInvalid())
      return true;
    Expr *E = Res.get();

    if (DI.isDeref()) {
      // Dereference the loaded pointer.
      auto Res =
          SemaRef.CreateBuiltinUnaryOp(RetVal->getBeginLoc(), UO_Deref, E);
      if (Res.isInvalid())
        return true;
      E = Res.get();
    }

    auto *OVE = OpaqueValueExpr::Wrap(SemaRef.Context, E);
    OVEs.push_back(OVE);

    unsigned Level = DI.isDeref() ? 1 : 0;
    Builder.insertDeclToNewValue(VD, OVE, Level);
  }

  auto Res = Builder.build(RetTy, RetOVE, /*WrappedValue=*/nullptr, &OVEs);
  if (Res.isInvalid())
    return true;

  S->setRetValue(Res.get());
  getParentMap().setParent(Res.get(), S);
  return false;
}

/// checkCountAttributedLocalsInBlock - This checks if dependent local variables
/// are declared within the same basic block.
bool checkCountAttributedLocalsInBlock(Sema &SemaRef, const CFGBlock *block) {
  llvm::SmallPtrSet<const Decl *, 4> LocalDeclSet;
  bool HadError = false;

  auto checkLocalDependentCountTypes = [&](const VarDecl *VD) {
    QualType Ty = VD->getType();
    while (!Ty.isNull()) {
      if (auto DCPTy = Ty->getAs<CountAttributedType>()) {
        for (const TypeCoupledDeclRefInfo &DI : DCPTy->dependent_decls()) {
          if (IsModifiableValue(SemaRef, DI.getDecl()) &&
              LocalDeclSet.find(DI.getDecl()) == LocalDeclSet.end()) {
            // The dependent decl can still be non-local variable if there was
            // an erroneous condition. Skip it to avoid reporting an irrelevant
            // error message.
            if (auto *VD = dyn_cast<VarDecl>(DI.getDecl())) {
              if (!VD->isLocalVarDecl()) {
                assert(SemaRef.hasUncompilableErrorOccurred());
                continue;
              }
              SemaRef.Diag(VD->getLocation(),
                           diag::err_bounds_safety_local_dependent_count_block);
              HadError = true;
            }
          }
        }
      }
      Ty = Ty->getPointeeType();
    }
  };

  for (CFGBlock::const_iterator BI = block->begin(), BE = block->end();
       BI != BE; ++BI) {
    if (std::optional<CFGStmt> stmt = BI->getAs<CFGStmt>()) {
      const DeclStmt *DS = dyn_cast<DeclStmt>(stmt->getStmt());
      if (!DS)
        continue;
      for (auto D : DS->decls()) {
        auto VD = dyn_cast<VarDecl>(D);
        if (!VD)
          continue;
        checkLocalDependentCountTypes(VD);
        LocalDeclSet.insert(VD);
      }
    }
  }
  return !HadError;
}

namespace clang {

void DynamicCountPointerAssignmentAnalysis::run() {
  AnalysisDeclContext AC(/* AnalysisDeclContextManager */ nullptr, dcl);

  CFG *cfg = AC.getCFG();
  if (!cfg)
    return;

  for (CFG::const_iterator I = cfg->begin(), E = cfg->end(); I != E; ++I) {
    const CFGBlock *block = *I;
    checkCountAttributedLocalsInBlock(SemaRef, block);
    CheckCountAttributedDeclAssignments checkDependents(SemaRef, AC);
    /// Visiting Stmt within a CFGBlock backward. Doing so makes easier to
    /// determine whether a function call is used in RHS of a tracked assignment
    /// expression or the call is standalone.
    for (CFGBlock::const_reverse_iterator BI = (*I)->rbegin(),
                                          BE = (*I)->rend();
         BI != BE; ++BI) {
      if (std::optional<CFGStmt> stmt = BI->getAs<CFGStmt>())
        checkDependents.BeginStmt(const_cast<Stmt *>(stmt->getStmt()));
    }
  }
}

RecordDecl *FlexibleArrayMemberUtils::GetFlexibleRecord(QualType QT) {
  auto *RT = QT->getAs<RecordType>();
  if (!RT)
    return nullptr;

  auto *RD = RT->getDecl();
  if (!RD->hasFlexibleArrayMember())
    return nullptr;

  const CountAttributedType *DCPTy = nullptr;
  RecordDecl *FlexibleRecord = RD;
  do {
    if (FlexibleRecord->getTagKind() == TagTypeKind::Union)
      return nullptr;
    FieldDecl *FlexibleField = nullptr;
    for (FieldDecl *FD : FlexibleRecord->fields()) {
      FlexibleField = FD;
    }
    assert(FlexibleField);
    QualType FieldType = FlexibleField->getType();
    if (auto *RT = FieldType->getAs<RecordType>()) {
      FlexibleRecord = RT->getDecl();
    } else {
      DCPTy = FieldType->isIncompleteArrayType() ? FieldType->getAs<CountAttributedType>() : nullptr;
      FlexibleRecord = nullptr;
    }
  } while (!DCPTy && FlexibleRecord);

  if (!DCPTy)
    return nullptr;

  return RD;
}

bool FlexibleArrayMemberUtils::Find(
    RecordDecl *RD, SmallVectorImpl<FieldDecl *> &PathToFlex,
    ArrayRef<TypeCoupledDeclRefInfo> &CountDecls) {
  if (!RD->hasFlexibleArrayMember())
    return false;

  const CountAttributedType *DCPTy = nullptr;
  RecordDecl *FlexibleRecord = RD;
  do {
    if (FlexibleRecord->getTagKind() == TagTypeKind::Union)
      return false;
    FieldDecl *FlexibleField = nullptr;
    for (FieldDecl *FD : FlexibleRecord->fields()) {
      FlexibleField = FD;
    }
    assert(FlexibleField);
    PathToFlex.push_back(FlexibleField);
    QualType FieldType = FlexibleField->getType();
    if (auto *RT = FieldType->getAs<RecordType>()) {
      FlexibleRecord = RT->getDecl();
    } else {
      DCPTy = FieldType->isIncompleteArrayType() ? FieldType->getAs<CountAttributedType>() : nullptr;
      FlexibleRecord = nullptr;
    }
  } while (!DCPTy && FlexibleRecord);

  if (!DCPTy)
    return false;

  CountDecls = DCPTy->getCoupledDecls();
  return true;
}

Expr *FlexibleArrayMemberUtils::SelectFlexibleObject(
    const SmallVectorImpl<FieldDecl *> &PathToFlex, Expr *Base) {
  bool IsArrow = Base->getType()->isPointerType();
  Expr *FlexibleObj = Base;
  for (unsigned I = 0; I < PathToFlex.size() - 1; ++I) {
    if (IsArrow && !FlexibleObj->isPRValue()) {
      FlexibleObj = ImplicitCastExpr::Create(
          SemaRef.Context, FlexibleObj->getType(), CK_LValueToRValue,
          FlexibleObj, nullptr, VK_PRValue, SemaRef.CurFPFeatureOverrides());
    }
    auto *FlexibleField = PathToFlex[I];
    FlexibleObj = MemberExpr::CreateImplicit(
        SemaRef.Context, FlexibleObj, IsArrow, FlexibleField,
        FlexibleField->getType(), VK_LValue, OK_Ordinary);
    IsArrow = false;
  }
  return FlexibleObj;
}

ExprResult FlexibleArrayMemberUtils::BuildCountExpr(
    FieldDecl *FlexibleField, const ArrayRef<TypeCoupledDeclRefInfo> CountDecls,
    Expr *StructBase, SmallVectorImpl<OpaqueValueExpr *> &OVEs,
    CopyExpr *DeclReplacer) {
  QualType FlexType = FlexibleField->getType();
  const auto *DCPTy = FlexType->getAs<CountAttributedType>();
  assert(DCPTy);
  Expr *CountTemplate = DCPTy->getCountExpr();

  CopyExpr Copy = DeclReplacer ? *DeclReplacer : CopyExpr(SemaRef);
  CopyExpr CopyNoRepl(SemaRef);
  bool IsArrow = StructBase->getType()->isPointerType();
  for (auto DeclRefInfo : CountDecls) {
    auto *VD = DeclRefInfo.getDecl();
    // For `counted_by(hdr.len)` we don't need a replacement for the MemberExpr
    // (`.len`), only the DeclRefExpr (`hdr`).
    // For `counted_by(len + len)` we don't need to add `len` twice - the full
    // count expression is transformed outside the loop and will replace both
    // DeclRefs regardless.
    if (DeclRefInfo.isMember() || Copy.HasDeclReplacement(VD))
      continue;

    if (auto *FD = dyn_cast<FieldDecl>(VD)) {
      Expr *Base = CopyNoRepl.TransformExpr(StructBase).get();
      Base = SemaRef.DefaultLvalueConversion(Base).get();
      if (IsArrow && !Base->isPRValue()) {
        Base = ImplicitCastExpr::Create(
            SemaRef.Context, Base->getType(), CK_LValueToRValue, Base, nullptr,
            VK_PRValue, SemaRef.CurFPFeatureOverrides());
      }
      ExprObjectKind OK = FD->isBitField() ? OK_BitField : OK_Ordinary;
      auto *NewMember = MemberExpr::CreateImplicit(
          SemaRef.Context, Base, IsArrow, FD, FD->getType(), VK_LValue, OK);
      ExprResult Lvalue = SemaRef.DefaultLvalueConversion(NewMember);
      if (!Lvalue.get())
        return ExprError();

      // Don't wrap the count member access in OVE. Instead, the user of
      // this expression should make sure it's always rebuilt. The base pointer
      // may be null so we guard member access to the base in the null check.
      // However, a materialization expression that wraps the null check may
      // still evaluate the member access, To avoid this, we do not wrap the
      // count member access in OVE.
      Copy.UnsafelyAddDeclSubstitution(FD, Lvalue.get());
    }
  }

  ExprResult Transformed = Copy.TransformExpr(CountTemplate);
  if (!Transformed.get())
    return ExprError();

  return SemaRef.DefaultLvalueConversion(Transformed.get());
}

TransformDynamicCountWithFunctionArgument ::
    TransformDynamicCountWithFunctionArgument(
        Sema &SemaRef, const SmallVectorImpl<Expr *> &Args, unsigned FirstParam)
    : BaseTransform(SemaRef), ActualArgs(Args), FirstParam(FirstParam) {}

ExprResult TransformDynamicCountWithFunctionArgument::TransformDeclRefExpr(
    DeclRefExpr *E) {
  ParmVarDecl *FormalArg = dyn_cast<ParmVarDecl>(E->getDecl());
  if (!FormalArg)
    return Owned(E);

  unsigned Index = FormalArg->getFunctionScopeIndex();
  assert(Index >= FirstParam);
  Index -= FirstParam;
  assert(Index < ActualArgs.size());
  return ActualArgs[Index];
}

} // namespace clang
