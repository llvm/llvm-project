//===----- SemaTypeTraits.cpp - Semantic Analysis for C++ Type Traits -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ type traits.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"
#include "clang/Basic/DiagnosticParse.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/EnterExpressionEvaluationContext.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaHLSL.h"

using namespace clang;

static CXXMethodDecl *LookupSpecialMemberFromXValue(Sema &SemaRef,
                                                    const CXXRecordDecl *RD,
                                                    bool Assign) {
  RD = RD->getDefinition();
  SourceLocation LookupLoc = RD->getLocation();

  CanQualType CanTy = SemaRef.getASTContext().getCanonicalType(
      SemaRef.getASTContext().getTagDeclType(RD));
  DeclarationName Name;
  Expr *Arg = nullptr;
  unsigned NumArgs;

  QualType ArgType = CanTy;
  ExprValueKind VK = clang::VK_XValue;

  if (Assign)
    Name =
        SemaRef.getASTContext().DeclarationNames.getCXXOperatorName(OO_Equal);
  else
    Name =
        SemaRef.getASTContext().DeclarationNames.getCXXConstructorName(CanTy);

  OpaqueValueExpr FakeArg(LookupLoc, ArgType, VK);
  NumArgs = 1;
  Arg = &FakeArg;

  // Create the object argument
  QualType ThisTy = CanTy;
  Expr::Classification Classification =
      OpaqueValueExpr(LookupLoc, ThisTy, VK_LValue)
          .Classify(SemaRef.getASTContext());

  // Now we perform lookup on the name we computed earlier and do overload
  // resolution. Lookup is only performed directly into the class since there
  // will always be a (possibly implicit) declaration to shadow any others.
  OverloadCandidateSet OCS(LookupLoc, OverloadCandidateSet::CSK_Normal);
  DeclContext::lookup_result R = RD->lookup(Name);

  if (R.empty())
    return nullptr;

  // Copy the candidates as our processing of them may load new declarations
  // from an external source and invalidate lookup_result.
  SmallVector<NamedDecl *, 8> Candidates(R.begin(), R.end());

  for (NamedDecl *CandDecl : Candidates) {
    if (CandDecl->isInvalidDecl())
      continue;

    DeclAccessPair Cand = DeclAccessPair::make(CandDecl, clang::AS_none);
    auto CtorInfo = getConstructorInfo(Cand);
    if (CXXMethodDecl *M = dyn_cast<CXXMethodDecl>(Cand->getUnderlyingDecl())) {
      if (Assign)
        SemaRef.AddMethodCandidate(M, Cand, const_cast<CXXRecordDecl *>(RD),
                                   ThisTy, Classification,
                                   llvm::ArrayRef(&Arg, NumArgs), OCS, true);
      else {
        assert(CtorInfo);
        SemaRef.AddOverloadCandidate(CtorInfo.Constructor, CtorInfo.FoundDecl,
                                     llvm::ArrayRef(&Arg, NumArgs), OCS,
                                     /*SuppressUserConversions*/ true);
      }
    } else if (FunctionTemplateDecl *Tmpl =
                   dyn_cast<FunctionTemplateDecl>(Cand->getUnderlyingDecl())) {
      if (Assign)
        SemaRef.AddMethodTemplateCandidate(
            Tmpl, Cand, const_cast<CXXRecordDecl *>(RD), nullptr, ThisTy,
            Classification, llvm::ArrayRef(&Arg, NumArgs), OCS, true);
      else {
        assert(CtorInfo);
        SemaRef.AddTemplateOverloadCandidate(
            CtorInfo.ConstructorTmpl, CtorInfo.FoundDecl, nullptr,
            llvm::ArrayRef(&Arg, NumArgs), OCS, true);
      }
    }
  }

  OverloadCandidateSet::iterator Best;
  switch (OCS.BestViableFunction(SemaRef, LookupLoc, Best)) {
  case OR_Success:
    return cast<CXXMethodDecl>(Best->Function);
  default:
    return nullptr;
  }
}

static bool hasSuitableConstructorForRelocation(Sema &SemaRef,
                                                const CXXRecordDecl *D,
                                                bool AllowUserDefined) {
  assert(D->hasDefinition() && !D->isInvalidDecl());

  if (D->hasSimpleMoveConstructor() || D->hasSimpleCopyConstructor())
    return true;

  CXXMethodDecl *Decl =
      LookupSpecialMemberFromXValue(SemaRef, D, /*Assign=*/false);
  return Decl && Decl->isUserProvided() == AllowUserDefined;
}

static bool hasSuitableMoveAssignmentOperatorForRelocation(
    Sema &SemaRef, const CXXRecordDecl *D, bool AllowUserDefined) {
  assert(D->hasDefinition() && !D->isInvalidDecl());

  if (D->hasSimpleMoveAssignment() || D->hasSimpleCopyAssignment())
    return true;

  CXXMethodDecl *Decl =
      LookupSpecialMemberFromXValue(SemaRef, D, /*Assign=*/true);
  if (!Decl)
    return false;

  return Decl && Decl->isUserProvided() == AllowUserDefined;
}

// [C++26][class.prop]
// A class C is default-movable if
// - overload resolution for direct-initializing an object of type C
// from an xvalue of type C selects a constructor that is a direct member of C
// and is neither user-provided nor deleted,
// - overload resolution for assigning to an lvalue of type C from an xvalue of
// type C selects an assignment operator function that is a direct member of C
// and is neither user-provided nor deleted, and C has a destructor that is
// neither user-provided nor deleted.
static bool IsDefaultMovable(Sema &SemaRef, const CXXRecordDecl *D) {
  if (!hasSuitableConstructorForRelocation(SemaRef, D,
                                           /*AllowUserDefined=*/false))
    return false;

  if (!hasSuitableMoveAssignmentOperatorForRelocation(
          SemaRef, D, /*AllowUserDefined=*/false))
    return false;

  CXXDestructorDecl *Dtr = D->getDestructor();

  if (!Dtr)
    return true;

  if (Dtr->isUserProvided() && (!Dtr->isDefaulted() || Dtr->isDeleted()))
    return false;

  return !Dtr->isDeleted();
}

// [C++26][class.prop]
// A class is eligible for trivial relocation unless it...
static bool IsEligibleForTrivialRelocation(Sema &SemaRef,
                                           const CXXRecordDecl *D) {

  for (const CXXBaseSpecifier &B : D->bases()) {
    const auto *BaseDecl = B.getType()->getAsCXXRecordDecl();
    if (!BaseDecl)
      continue;
    // ... has any virtual base classes
    // ... has a base class that is not a trivially relocatable class
    if (B.isVirtual() || (!BaseDecl->isDependentType() &&
                          !SemaRef.IsCXXTriviallyRelocatableType(B.getType())))
      return false;
  }

  for (const FieldDecl *Field : D->fields()) {
    if (Field->getType()->isDependentType())
      continue;
    if (Field->getType()->isReferenceType())
      continue;
    // ... has a non-static data member of an object type that is not
    // of a trivially relocatable type
    if (!SemaRef.IsCXXTriviallyRelocatableType(Field->getType()))
      return false;
  }
  return !D->hasDeletedDestructor();
}

// [C++26][class.prop]
// A class C is eligible for replacement unless
static bool IsEligibleForReplacement(Sema &SemaRef, const CXXRecordDecl *D) {

  for (const CXXBaseSpecifier &B : D->bases()) {
    const auto *BaseDecl = B.getType()->getAsCXXRecordDecl();
    if (!BaseDecl)
      continue;
    // it has a base class that is not a replaceable class
    if (!BaseDecl->isDependentType() &&
        !SemaRef.IsCXXReplaceableType(B.getType()))
      return false;
  }

  for (const FieldDecl *Field : D->fields()) {
    if (Field->getType()->isDependentType())
      continue;

    // it has a non-static data member that is not of a replaceable type,
    if (!SemaRef.IsCXXReplaceableType(Field->getType()))
      return false;
  }
  return !D->hasDeletedDestructor();
}

ASTContext::CXXRecordDeclRelocationInfo
Sema::CheckCXX2CRelocatableAndReplaceable(const CXXRecordDecl *D) {
  ASTContext::CXXRecordDeclRelocationInfo Info{false, false};

  if (!getLangOpts().CPlusPlus || D->isInvalidDecl())
    return Info;

  assert(D->hasDefinition());

  // This is part of "eligible for replacement", however we defer it
  // to avoid extraneous computations.
  auto HasSuitableSMP = [&] {
    return hasSuitableConstructorForRelocation(*this, D,
                                               /*AllowUserDefined=*/true) &&
           hasSuitableMoveAssignmentOperatorForRelocation(
               *this, D, /*AllowUserDefined=*/true);
  };

  auto IsUnion = [&, Is = std::optional<bool>{}]() mutable {
    if (!Is.has_value())
      Is = D->isUnion() && !D->hasUserDeclaredCopyConstructor() &&
           !D->hasUserDeclaredCopyAssignment() &&
           !D->hasUserDeclaredMoveOperation() &&
           !D->hasUserDeclaredDestructor();
    return *Is;
  };

  auto IsDefaultMovable = [&, Is = std::optional<bool>{}]() mutable {
    if (!Is.has_value())
      Is = ::IsDefaultMovable(*this, D);
    return *Is;
  };

  Info.IsRelocatable = [&] {
    if (D->isDependentType())
      return false;

    // if it is eligible for trivial relocation
    if (!IsEligibleForTrivialRelocation(*this, D))
      return false;

    // has the trivially_relocatable_if_eligible class-property-specifier,
    if (D->hasAttr<TriviallyRelocatableAttr>())
      return true;

    // is a union with no user-declared special member functions, or
    if (IsUnion())
      return true;

    // is default-movable.
    return IsDefaultMovable();
  }();

  Info.IsReplaceable = [&] {
    if (D->isDependentType())
      return false;

    // A class C is a replaceable class if it is eligible for replacement
    if (!IsEligibleForReplacement(*this, D))
      return false;

    // has the replaceable_if_eligible class-property-specifier
    if (D->hasAttr<ReplaceableAttr>())
      return HasSuitableSMP();

    // is a union with no user-declared special member functions, or
    if (IsUnion())
      return HasSuitableSMP();

    // is default-movable.
    return IsDefaultMovable();
  }();

  return Info;
}

static bool IsCXXTriviallyRelocatableType(Sema &S, const CXXRecordDecl *RD) {
  if (std::optional<ASTContext::CXXRecordDeclRelocationInfo> Info =
          S.getASTContext().getRelocationInfoForCXXRecord(RD))
    return Info->IsRelocatable;
  ASTContext::CXXRecordDeclRelocationInfo Info =
      S.CheckCXX2CRelocatableAndReplaceable(RD);
  S.getASTContext().setRelocationInfoForCXXRecord(RD, Info);
  return Info.IsRelocatable;
}

bool Sema::IsCXXTriviallyRelocatableType(QualType Type) {

  QualType BaseElementType = getASTContext().getBaseElementType(Type);

  if (Type->isVariableArrayType())
    return false;

  if (BaseElementType.hasNonTrivialObjCLifetime())
    return false;

  if (BaseElementType.hasAddressDiscriminatedPointerAuth())
    return false;

  if (BaseElementType->isIncompleteType())
    return false;

  if (BaseElementType->isScalarType() || BaseElementType->isVectorType())
    return true;

  if (const auto *RD = BaseElementType->getAsCXXRecordDecl())
    return ::IsCXXTriviallyRelocatableType(*this, RD);

  return false;
}

static bool IsCXXReplaceableType(Sema &S, const CXXRecordDecl *RD) {
  if (std::optional<ASTContext::CXXRecordDeclRelocationInfo> Info =
          S.getASTContext().getRelocationInfoForCXXRecord(RD))
    return Info->IsReplaceable;
  ASTContext::CXXRecordDeclRelocationInfo Info =
      S.CheckCXX2CRelocatableAndReplaceable(RD);
  S.getASTContext().setRelocationInfoForCXXRecord(RD, Info);
  return Info.IsReplaceable;
}

bool Sema::IsCXXReplaceableType(QualType Type) {
  if (Type.isConstQualified() || Type.isVolatileQualified())
    return false;

  if (Type->isVariableArrayType())
    return false;

  QualType BaseElementType =
      getASTContext().getBaseElementType(Type.getUnqualifiedType());
  if (BaseElementType->isIncompleteType())
    return false;
  if (BaseElementType->isScalarType())
    return true;
  if (const auto *RD = BaseElementType->getAsCXXRecordDecl())
    return ::IsCXXReplaceableType(*this, RD);
  return false;
}

/// Checks that type T is not a VLA.
///
/// @returns @c true if @p T is VLA and a diagnostic was emitted,
/// @c false otherwise.
static bool DiagnoseVLAInCXXTypeTrait(Sema &S, const TypeSourceInfo *T,
                                      clang::tok::TokenKind TypeTraitID) {
  if (!T->getType()->isVariableArrayType())
    return false;

  S.Diag(T->getTypeLoc().getBeginLoc(), diag::err_vla_unsupported)
      << 1 << TypeTraitID;
  return true;
}

/// Checks that type T is not an atomic type (_Atomic).
///
/// @returns @c true if @p T is VLA and a diagnostic was emitted,
/// @c false otherwise.
static bool DiagnoseAtomicInCXXTypeTrait(Sema &S, const TypeSourceInfo *T,
                                         clang::tok::TokenKind TypeTraitID) {
  if (!T->getType()->isAtomicType())
    return false;

  S.Diag(T->getTypeLoc().getBeginLoc(), diag::err_atomic_unsupported)
      << TypeTraitID;
  return true;
}

/// Check the completeness of a type in a unary type trait.
///
/// If the particular type trait requires a complete type, tries to complete
/// it. If completing the type fails, a diagnostic is emitted and false
/// returned. If completing the type succeeds or no completion was required,
/// returns true.
static bool CheckUnaryTypeTraitTypeCompleteness(Sema &S, TypeTrait UTT,
                                                SourceLocation Loc,
                                                QualType ArgTy) {
  // C++0x [meta.unary.prop]p3:
  //   For all of the class templates X declared in this Clause, instantiating
  //   that template with a template argument that is a class template
  //   specialization may result in the implicit instantiation of the template
  //   argument if and only if the semantics of X require that the argument
  //   must be a complete type.
  // We apply this rule to all the type trait expressions used to implement
  // these class templates. We also try to follow any GCC documented behavior
  // in these expressions to ensure portability of standard libraries.
  switch (UTT) {
  default:
    llvm_unreachable("not a UTT");
    // is_complete_type somewhat obviously cannot require a complete type.
  case UTT_IsCompleteType:
    // Fall-through

    // These traits are modeled on the type predicates in C++0x
    // [meta.unary.cat] and [meta.unary.comp]. They are not specified as
    // requiring a complete type, as whether or not they return true cannot be
    // impacted by the completeness of the type.
  case UTT_IsVoid:
  case UTT_IsIntegral:
  case UTT_IsFloatingPoint:
  case UTT_IsArray:
  case UTT_IsBoundedArray:
  case UTT_IsPointer:
  case UTT_IsLvalueReference:
  case UTT_IsRvalueReference:
  case UTT_IsMemberFunctionPointer:
  case UTT_IsMemberObjectPointer:
  case UTT_IsEnum:
  case UTT_IsScopedEnum:
  case UTT_IsUnion:
  case UTT_IsClass:
  case UTT_IsFunction:
  case UTT_IsReference:
  case UTT_IsArithmetic:
  case UTT_IsFundamental:
  case UTT_IsObject:
  case UTT_IsScalar:
  case UTT_IsCompound:
  case UTT_IsMemberPointer:
  case UTT_IsTypedResourceElementCompatible:
    // Fall-through

    // These traits are modeled on type predicates in C++0x [meta.unary.prop]
    // which requires some of its traits to have the complete type. However,
    // the completeness of the type cannot impact these traits' semantics, and
    // so they don't require it. This matches the comments on these traits in
    // Table 49.
  case UTT_IsConst:
  case UTT_IsVolatile:
  case UTT_IsSigned:
  case UTT_IsUnboundedArray:
  case UTT_IsUnsigned:

    // This type trait always returns false, checking the type is moot.
  case UTT_IsInterfaceClass:
    return true;

    // We diagnose incomplete class types later.
  case UTT_StructuredBindingSize:
    return true;

    // C++14 [meta.unary.prop]:
    //   If T is a non-union class type, T shall be a complete type.
  case UTT_IsEmpty:
  case UTT_IsPolymorphic:
  case UTT_IsAbstract:
    if (const auto *RD = ArgTy->getAsCXXRecordDecl())
      if (!RD->isUnion())
        return !S.RequireCompleteType(
            Loc, ArgTy, diag::err_incomplete_type_used_in_type_trait_expr);
    return true;

    // C++14 [meta.unary.prop]:
    //   If T is a class type, T shall be a complete type.
  case UTT_IsFinal:
  case UTT_IsSealed:
    if (ArgTy->getAsCXXRecordDecl())
      return !S.RequireCompleteType(
          Loc, ArgTy, diag::err_incomplete_type_used_in_type_trait_expr);
    return true;

    // LWG3823: T shall be an array type, a complete type, or cv void.
  case UTT_IsAggregate:
  case UTT_IsImplicitLifetime:
    if (ArgTy->isArrayType() || ArgTy->isVoidType())
      return true;

    return !S.RequireCompleteType(
        Loc, ArgTy, diag::err_incomplete_type_used_in_type_trait_expr);

    // has_unique_object_representations<T>
    // remove_all_extents_t<T> shall be a complete type or cv void (LWG4113).
  case UTT_HasUniqueObjectRepresentations:
    ArgTy = QualType(ArgTy->getBaseElementTypeUnsafe(), 0);
    if (ArgTy->isVoidType())
      return true;
    return !S.RequireCompleteType(
        Loc, ArgTy, diag::err_incomplete_type_used_in_type_trait_expr);

    // C++1z [meta.unary.prop]:
    //   remove_all_extents_t<T> shall be a complete type or cv void.
  case UTT_IsTrivial:
  case UTT_IsTriviallyCopyable:
  case UTT_IsStandardLayout:
  case UTT_IsPOD:
  case UTT_IsLiteral:
  case UTT_IsBitwiseCloneable:
  // By analogy, is_trivially_relocatable and is_trivially_equality_comparable
  // impose the same constraints.
  case UTT_IsTriviallyRelocatable:
  case UTT_IsTriviallyEqualityComparable:
  case UTT_IsCppTriviallyRelocatable:
  case UTT_IsReplaceable:
  case UTT_CanPassInRegs:
  // Per the GCC type traits documentation, T shall be a complete type, cv void,
  // or an array of unknown bound. But GCC actually imposes the same constraints
  // as above.
  case UTT_HasNothrowAssign:
  case UTT_HasNothrowMoveAssign:
  case UTT_HasNothrowConstructor:
  case UTT_HasNothrowCopy:
  case UTT_HasTrivialAssign:
  case UTT_HasTrivialMoveAssign:
  case UTT_HasTrivialDefaultConstructor:
  case UTT_HasTrivialMoveConstructor:
  case UTT_HasTrivialCopy:
  case UTT_HasTrivialDestructor:
  case UTT_HasVirtualDestructor:
    ArgTy = QualType(ArgTy->getBaseElementTypeUnsafe(), 0);
    [[fallthrough]];
  // C++1z [meta.unary.prop]:
  //   T shall be a complete type, cv void, or an array of unknown bound.
  case UTT_IsDestructible:
  case UTT_IsNothrowDestructible:
  case UTT_IsTriviallyDestructible:
  case UTT_IsIntangibleType:
    if (ArgTy->isIncompleteArrayType() || ArgTy->isVoidType())
      return true;

    return !S.RequireCompleteType(
        Loc, ArgTy, diag::err_incomplete_type_used_in_type_trait_expr);
  }
}

static bool HasNoThrowOperator(const RecordType *RT, OverloadedOperatorKind Op,
                               Sema &Self, SourceLocation KeyLoc, ASTContext &C,
                               bool (CXXRecordDecl::*HasTrivial)() const,
                               bool (CXXRecordDecl::*HasNonTrivial)() const,
                               bool (CXXMethodDecl::*IsDesiredOp)() const) {
  CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
  if ((RD->*HasTrivial)() && !(RD->*HasNonTrivial)())
    return true;

  DeclarationName Name = C.DeclarationNames.getCXXOperatorName(Op);
  DeclarationNameInfo NameInfo(Name, KeyLoc);
  LookupResult Res(Self, NameInfo, Sema::LookupOrdinaryName);
  if (Self.LookupQualifiedName(Res, RD)) {
    bool FoundOperator = false;
    Res.suppressDiagnostics();
    for (LookupResult::iterator Op = Res.begin(), OpEnd = Res.end();
         Op != OpEnd; ++Op) {
      if (isa<FunctionTemplateDecl>(*Op))
        continue;

      CXXMethodDecl *Operator = cast<CXXMethodDecl>(*Op);
      if ((Operator->*IsDesiredOp)()) {
        FoundOperator = true;
        auto *CPT = Operator->getType()->castAs<FunctionProtoType>();
        CPT = Self.ResolveExceptionSpec(KeyLoc, CPT);
        if (!CPT || !CPT->isNothrow())
          return false;
      }
    }
    return FoundOperator;
  }
  return false;
}

static bool HasNonDeletedDefaultedEqualityComparison(Sema &S,
                                                     const CXXRecordDecl *Decl,
                                                     SourceLocation KeyLoc) {
  if (Decl->isUnion())
    return false;
  if (Decl->isLambda())
    return Decl->isCapturelessLambda();

  {
    EnterExpressionEvaluationContext UnevaluatedContext(
        S, Sema::ExpressionEvaluationContext::Unevaluated);
    Sema::SFINAETrap SFINAE(S, /*ForValidityCheck=*/true);
    Sema::ContextRAII TUContext(S, S.Context.getTranslationUnitDecl());

    // const ClassT& obj;
    OpaqueValueExpr Operand(
        KeyLoc,
        Decl->getTypeForDecl()->getCanonicalTypeUnqualified().withConst(),
        ExprValueKind::VK_LValue);
    UnresolvedSet<16> Functions;
    // obj == obj;
    S.LookupBinOp(S.TUScope, {}, BinaryOperatorKind::BO_EQ, Functions);

    auto Result = S.CreateOverloadedBinOp(KeyLoc, BinaryOperatorKind::BO_EQ,
                                          Functions, &Operand, &Operand);
    if (Result.isInvalid() || SFINAE.hasErrorOccurred())
      return false;

    const auto *CallExpr = dyn_cast<CXXOperatorCallExpr>(Result.get());
    if (!CallExpr)
      return false;
    const auto *Callee = CallExpr->getDirectCallee();
    auto ParamT = Callee->getParamDecl(0)->getType();
    if (!Callee->isDefaulted())
      return false;
    if (!ParamT->isReferenceType() && !Decl->isTriviallyCopyable())
      return false;
    if (ParamT.getNonReferenceType()->getUnqualifiedDesugaredType() !=
        Decl->getTypeForDecl())
      return false;
  }

  return llvm::all_of(Decl->bases(),
                      [&](const CXXBaseSpecifier &BS) {
                        if (const auto *RD = BS.getType()->getAsCXXRecordDecl())
                          return HasNonDeletedDefaultedEqualityComparison(
                              S, RD, KeyLoc);
                        return true;
                      }) &&
         llvm::all_of(Decl->fields(), [&](const FieldDecl *FD) {
           auto Type = FD->getType();
           if (Type->isArrayType())
             Type = Type->getBaseElementTypeUnsafe()
                        ->getCanonicalTypeUnqualified();

           if (Type->isReferenceType() || Type->isEnumeralType())
             return false;
           if (const auto *RD = Type->getAsCXXRecordDecl())
             return HasNonDeletedDefaultedEqualityComparison(S, RD, KeyLoc);
           return true;
         });
}

static bool isTriviallyEqualityComparableType(Sema &S, QualType Type,
                                              SourceLocation KeyLoc) {
  QualType CanonicalType = Type.getCanonicalType();
  if (CanonicalType->isIncompleteType() || CanonicalType->isDependentType() ||
      CanonicalType->isEnumeralType() || CanonicalType->isArrayType())
    return false;

  if (const auto *RD = CanonicalType->getAsCXXRecordDecl()) {
    if (!HasNonDeletedDefaultedEqualityComparison(S, RD, KeyLoc))
      return false;
  }

  return S.getASTContext().hasUniqueObjectRepresentations(
      CanonicalType, /*CheckIfTriviallyCopyable=*/false);
}

static bool IsTriviallyRelocatableType(Sema &SemaRef, QualType T) {
  QualType BaseElementType = SemaRef.getASTContext().getBaseElementType(T);

  if (BaseElementType->isIncompleteType())
    return false;
  if (!BaseElementType->isObjectType())
    return false;

  if (T.hasAddressDiscriminatedPointerAuth())
    return false;

  if (const auto *RD = BaseElementType->getAsCXXRecordDecl();
      RD && !RD->isPolymorphic() && IsCXXTriviallyRelocatableType(SemaRef, RD))
    return true;

  if (const auto *RD = BaseElementType->getAsRecordDecl())
    return RD->canPassInRegisters();

  if (BaseElementType.isTriviallyCopyableType(SemaRef.getASTContext()))
    return true;

  switch (T.isNonTrivialToPrimitiveDestructiveMove()) {
  case QualType::PCK_Trivial:
    return !T.isDestructedType();
  case QualType::PCK_ARCStrong:
    return true;
  default:
    return false;
  }
}

static bool EvaluateUnaryTypeTrait(Sema &Self, TypeTrait UTT,
                                   SourceLocation KeyLoc,
                                   TypeSourceInfo *TInfo) {
  QualType T = TInfo->getType();
  assert(!T->isDependentType() && "Cannot evaluate traits of dependent type");

  ASTContext &C = Self.Context;
  switch (UTT) {
  default:
    llvm_unreachable("not a UTT");
    // Type trait expressions corresponding to the primary type category
    // predicates in C++0x [meta.unary.cat].
  case UTT_IsVoid:
    return T->isVoidType();
  case UTT_IsIntegral:
    return T->isIntegralType(C);
  case UTT_IsFloatingPoint:
    return T->isFloatingType();
  case UTT_IsArray:
    // Zero-sized arrays aren't considered arrays in partial specializations,
    // so __is_array shouldn't consider them arrays either.
    if (const auto *CAT = C.getAsConstantArrayType(T))
      return CAT->getSize() != 0;
    return T->isArrayType();
  case UTT_IsBoundedArray:
    if (DiagnoseVLAInCXXTypeTrait(Self, TInfo, tok::kw___is_bounded_array))
      return false;
    // Zero-sized arrays aren't considered arrays in partial specializations,
    // so __is_bounded_array shouldn't consider them arrays either.
    if (const auto *CAT = C.getAsConstantArrayType(T))
      return CAT->getSize() != 0;
    return T->isArrayType() && !T->isIncompleteArrayType();
  case UTT_IsUnboundedArray:
    if (DiagnoseVLAInCXXTypeTrait(Self, TInfo, tok::kw___is_unbounded_array))
      return false;
    return T->isIncompleteArrayType();
  case UTT_IsPointer:
    return T->isAnyPointerType();
  case UTT_IsLvalueReference:
    return T->isLValueReferenceType();
  case UTT_IsRvalueReference:
    return T->isRValueReferenceType();
  case UTT_IsMemberFunctionPointer:
    return T->isMemberFunctionPointerType();
  case UTT_IsMemberObjectPointer:
    return T->isMemberDataPointerType();
  case UTT_IsEnum:
    return T->isEnumeralType();
  case UTT_IsScopedEnum:
    return T->isScopedEnumeralType();
  case UTT_IsUnion:
    return T->isUnionType();
  case UTT_IsClass:
    return T->isClassType() || T->isStructureType() || T->isInterfaceType();
  case UTT_IsFunction:
    return T->isFunctionType();

    // Type trait expressions which correspond to the convenient composition
    // predicates in C++0x [meta.unary.comp].
  case UTT_IsReference:
    return T->isReferenceType();
  case UTT_IsArithmetic:
    return T->isArithmeticType() && !T->isEnumeralType();
  case UTT_IsFundamental:
    return T->isFundamentalType();
  case UTT_IsObject:
    return T->isObjectType();
  case UTT_IsScalar:
    // Note: semantic analysis depends on Objective-C lifetime types to be
    // considered scalar types. However, such types do not actually behave
    // like scalar types at run time (since they may require retain/release
    // operations), so we report them as non-scalar.
    if (T->isObjCLifetimeType()) {
      switch (T.getObjCLifetime()) {
      case Qualifiers::OCL_None:
      case Qualifiers::OCL_ExplicitNone:
        return true;

      case Qualifiers::OCL_Strong:
      case Qualifiers::OCL_Weak:
      case Qualifiers::OCL_Autoreleasing:
        return false;
      }
    }

    return T->isScalarType();
  case UTT_IsCompound:
    return T->isCompoundType();
  case UTT_IsMemberPointer:
    return T->isMemberPointerType();

    // Type trait expressions which correspond to the type property predicates
    // in C++0x [meta.unary.prop].
  case UTT_IsConst:
    return T.isConstQualified();
  case UTT_IsVolatile:
    return T.isVolatileQualified();
  case UTT_IsTrivial:
    return T.isTrivialType(C);
  case UTT_IsTriviallyCopyable:
    return T.isTriviallyCopyableType(C);
  case UTT_IsStandardLayout:
    return T->isStandardLayoutType();
  case UTT_IsPOD:
    return T.isPODType(C);
  case UTT_IsLiteral:
    return T->isLiteralType(C);
  case UTT_IsEmpty:
    if (const CXXRecordDecl *RD = T->getAsCXXRecordDecl())
      return !RD->isUnion() && RD->isEmpty();
    return false;
  case UTT_IsPolymorphic:
    if (const CXXRecordDecl *RD = T->getAsCXXRecordDecl())
      return !RD->isUnion() && RD->isPolymorphic();
    return false;
  case UTT_IsAbstract:
    if (const CXXRecordDecl *RD = T->getAsCXXRecordDecl())
      return !RD->isUnion() && RD->isAbstract();
    return false;
  case UTT_IsAggregate:
    // Report vector extensions and complex types as aggregates because they
    // support aggregate initialization. GCC mirrors this behavior for vectors
    // but not _Complex.
    return T->isAggregateType() || T->isVectorType() || T->isExtVectorType() ||
           T->isAnyComplexType();
  // __is_interface_class only returns true when CL is invoked in /CLR mode and
  // even then only when it is used with the 'interface struct ...' syntax
  // Clang doesn't support /CLR which makes this type trait moot.
  case UTT_IsInterfaceClass:
    return false;
  case UTT_IsFinal:
  case UTT_IsSealed:
    if (const CXXRecordDecl *RD = T->getAsCXXRecordDecl())
      return RD->hasAttr<FinalAttr>();
    return false;
  case UTT_IsSigned:
    // Enum types should always return false.
    // Floating points should always return true.
    return T->isFloatingType() ||
           (T->isSignedIntegerType() && !T->isEnumeralType());
  case UTT_IsUnsigned:
    // Enum types should always return false.
    return T->isUnsignedIntegerType() && !T->isEnumeralType();

    // Type trait expressions which query classes regarding their construction,
    // destruction, and copying. Rather than being based directly on the
    // related type predicates in the standard, they are specified by both
    // GCC[1] and the Embarcadero C++ compiler[2], and Clang implements those
    // specifications.
    //
    //   1: http://gcc.gnu/.org/onlinedocs/gcc/Type-Traits.html
    //   2:
    //   http://docwiki.embarcadero.com/RADStudio/XE/en/Type_Trait_Functions_(C%2B%2B0x)_Index
    //
    // Note that these builtins do not behave as documented in g++: if a class
    // has both a trivial and a non-trivial special member of a particular kind,
    // they return false! For now, we emulate this behavior.
    // FIXME: This appears to be a g++ bug: more complex cases reveal that it
    // does not correctly compute triviality in the presence of multiple special
    // members of the same kind. Revisit this once the g++ bug is fixed.
  case UTT_HasTrivialDefaultConstructor:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __is_pod (type) is true then the trait is true, else if type is
    //   a cv class or union type (or array thereof) with a trivial default
    //   constructor ([class.ctor]) then the trait is true, else it is false.
    if (T.isPODType(C))
      return true;
    if (CXXRecordDecl *RD = C.getBaseElementType(T)->getAsCXXRecordDecl())
      return RD->hasTrivialDefaultConstructor() &&
             !RD->hasNonTrivialDefaultConstructor();
    return false;
  case UTT_HasTrivialMoveConstructor:
    //  This trait is implemented by MSVC 2012 and needed to parse the
    //  standard library headers. Specifically this is used as the logic
    //  behind std::is_trivially_move_constructible (20.9.4.3).
    if (T.isPODType(C))
      return true;
    if (CXXRecordDecl *RD = C.getBaseElementType(T)->getAsCXXRecordDecl())
      return RD->hasTrivialMoveConstructor() &&
             !RD->hasNonTrivialMoveConstructor();
    return false;
  case UTT_HasTrivialCopy:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __is_pod (type) is true or type is a reference type then
    //   the trait is true, else if type is a cv class or union type
    //   with a trivial copy constructor ([class.copy]) then the trait
    //   is true, else it is false.
    if (T.isPODType(C) || T->isReferenceType())
      return true;
    if (CXXRecordDecl *RD = T->getAsCXXRecordDecl())
      return RD->hasTrivialCopyConstructor() &&
             !RD->hasNonTrivialCopyConstructor();
    return false;
  case UTT_HasTrivialMoveAssign:
    //  This trait is implemented by MSVC 2012 and needed to parse the
    //  standard library headers. Specifically it is used as the logic
    //  behind std::is_trivially_move_assignable (20.9.4.3)
    if (T.isPODType(C))
      return true;
    if (CXXRecordDecl *RD = C.getBaseElementType(T)->getAsCXXRecordDecl())
      return RD->hasTrivialMoveAssignment() &&
             !RD->hasNonTrivialMoveAssignment();
    return false;
  case UTT_HasTrivialAssign:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If type is const qualified or is a reference type then the
    //   trait is false. Otherwise if __is_pod (type) is true then the
    //   trait is true, else if type is a cv class or union type with
    //   a trivial copy assignment ([class.copy]) then the trait is
    //   true, else it is false.
    // Note: the const and reference restrictions are interesting,
    // given that const and reference members don't prevent a class
    // from having a trivial copy assignment operator (but do cause
    // errors if the copy assignment operator is actually used, q.v.
    // [class.copy]p12).

    if (T.isConstQualified())
      return false;
    if (T.isPODType(C))
      return true;
    if (CXXRecordDecl *RD = T->getAsCXXRecordDecl())
      return RD->hasTrivialCopyAssignment() &&
             !RD->hasNonTrivialCopyAssignment();
    return false;
  case UTT_IsDestructible:
  case UTT_IsTriviallyDestructible:
  case UTT_IsNothrowDestructible:
    // C++14 [meta.unary.prop]:
    //   For reference types, is_destructible<T>::value is true.
    if (T->isReferenceType())
      return true;

    // Objective-C++ ARC: autorelease types don't require destruction.
    if (T->isObjCLifetimeType() &&
        T.getObjCLifetime() == Qualifiers::OCL_Autoreleasing)
      return true;

    // C++14 [meta.unary.prop]:
    //   For incomplete types and function types, is_destructible<T>::value is
    //   false.
    if (T->isIncompleteType() || T->isFunctionType())
      return false;

    // A type that requires destruction (via a non-trivial destructor or ARC
    // lifetime semantics) is not trivially-destructible.
    if (UTT == UTT_IsTriviallyDestructible && T.isDestructedType())
      return false;

    // C++14 [meta.unary.prop]:
    //   For object types and given U equal to remove_all_extents_t<T>, if the
    //   expression std::declval<U&>().~U() is well-formed when treated as an
    //   unevaluated operand (Clause 5), then is_destructible<T>::value is true
    if (auto *RD = C.getBaseElementType(T)->getAsCXXRecordDecl()) {
      CXXDestructorDecl *Destructor = Self.LookupDestructor(RD);
      if (!Destructor)
        return false;
      //  C++14 [dcl.fct.def.delete]p2:
      //    A program that refers to a deleted function implicitly or
      //    explicitly, other than to declare it, is ill-formed.
      if (Destructor->isDeleted())
        return false;
      if (C.getLangOpts().AccessControl && Destructor->getAccess() != AS_public)
        return false;
      if (UTT == UTT_IsNothrowDestructible) {
        auto *CPT = Destructor->getType()->castAs<FunctionProtoType>();
        CPT = Self.ResolveExceptionSpec(KeyLoc, CPT);
        if (!CPT || !CPT->isNothrow())
          return false;
      }
    }
    return true;

  case UTT_HasTrivialDestructor:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html
    //   If __is_pod (type) is true or type is a reference type
    //   then the trait is true, else if type is a cv class or union
    //   type (or array thereof) with a trivial destructor
    //   ([class.dtor]) then the trait is true, else it is
    //   false.
    if (T.isPODType(C) || T->isReferenceType())
      return true;

    // Objective-C++ ARC: autorelease types don't require destruction.
    if (T->isObjCLifetimeType() &&
        T.getObjCLifetime() == Qualifiers::OCL_Autoreleasing)
      return true;

    if (CXXRecordDecl *RD = C.getBaseElementType(T)->getAsCXXRecordDecl())
      return RD->hasTrivialDestructor();
    return false;
  // TODO: Propagate nothrowness for implicitly declared special members.
  case UTT_HasNothrowAssign:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If type is const qualified or is a reference type then the
    //   trait is false. Otherwise if __has_trivial_assign (type)
    //   is true then the trait is true, else if type is a cv class
    //   or union type with copy assignment operators that are known
    //   not to throw an exception then the trait is true, else it is
    //   false.
    if (C.getBaseElementType(T).isConstQualified())
      return false;
    if (T->isReferenceType())
      return false;
    if (T.isPODType(C) || T->isObjCLifetimeType())
      return true;

    if (const RecordType *RT = T->getAs<RecordType>())
      return HasNoThrowOperator(RT, OO_Equal, Self, KeyLoc, C,
                                &CXXRecordDecl::hasTrivialCopyAssignment,
                                &CXXRecordDecl::hasNonTrivialCopyAssignment,
                                &CXXMethodDecl::isCopyAssignmentOperator);
    return false;
  case UTT_HasNothrowMoveAssign:
    //  This trait is implemented by MSVC 2012 and needed to parse the
    //  standard library headers. Specifically this is used as the logic
    //  behind std::is_nothrow_move_assignable (20.9.4.3).
    if (T.isPODType(C))
      return true;

    if (const RecordType *RT = C.getBaseElementType(T)->getAs<RecordType>())
      return HasNoThrowOperator(RT, OO_Equal, Self, KeyLoc, C,
                                &CXXRecordDecl::hasTrivialMoveAssignment,
                                &CXXRecordDecl::hasNonTrivialMoveAssignment,
                                &CXXMethodDecl::isMoveAssignmentOperator);
    return false;
  case UTT_HasNothrowCopy:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __has_trivial_copy (type) is true then the trait is true, else
    //   if type is a cv class or union type with copy constructors that are
    //   known not to throw an exception then the trait is true, else it is
    //   false.
    if (T.isPODType(C) || T->isReferenceType() || T->isObjCLifetimeType())
      return true;
    if (CXXRecordDecl *RD = T->getAsCXXRecordDecl()) {
      if (RD->hasTrivialCopyConstructor() &&
          !RD->hasNonTrivialCopyConstructor())
        return true;

      bool FoundConstructor = false;
      unsigned FoundTQs;
      for (const auto *ND : Self.LookupConstructors(RD)) {
        // A template constructor is never a copy constructor.
        // FIXME: However, it may actually be selected at the actual overload
        // resolution point.
        if (isa<FunctionTemplateDecl>(ND->getUnderlyingDecl()))
          continue;
        // UsingDecl itself is not a constructor
        if (isa<UsingDecl>(ND))
          continue;
        auto *Constructor = cast<CXXConstructorDecl>(ND->getUnderlyingDecl());
        if (Constructor->isCopyConstructor(FoundTQs)) {
          FoundConstructor = true;
          auto *CPT = Constructor->getType()->castAs<FunctionProtoType>();
          CPT = Self.ResolveExceptionSpec(KeyLoc, CPT);
          if (!CPT)
            return false;
          // TODO: check whether evaluating default arguments can throw.
          // For now, we'll be conservative and assume that they can throw.
          if (!CPT->isNothrow() || CPT->getNumParams() > 1)
            return false;
        }
      }

      return FoundConstructor;
    }
    return false;
  case UTT_HasNothrowConstructor:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html
    //   If __has_trivial_constructor (type) is true then the trait is
    //   true, else if type is a cv class or union type (or array
    //   thereof) with a default constructor that is known not to
    //   throw an exception then the trait is true, else it is false.
    if (T.isPODType(C) || T->isObjCLifetimeType())
      return true;
    if (CXXRecordDecl *RD = C.getBaseElementType(T)->getAsCXXRecordDecl()) {
      if (RD->hasTrivialDefaultConstructor() &&
          !RD->hasNonTrivialDefaultConstructor())
        return true;

      bool FoundConstructor = false;
      for (const auto *ND : Self.LookupConstructors(RD)) {
        // FIXME: In C++0x, a constructor template can be a default constructor.
        if (isa<FunctionTemplateDecl>(ND->getUnderlyingDecl()))
          continue;
        // UsingDecl itself is not a constructor
        if (isa<UsingDecl>(ND))
          continue;
        auto *Constructor = cast<CXXConstructorDecl>(ND->getUnderlyingDecl());
        if (Constructor->isDefaultConstructor()) {
          FoundConstructor = true;
          auto *CPT = Constructor->getType()->castAs<FunctionProtoType>();
          CPT = Self.ResolveExceptionSpec(KeyLoc, CPT);
          if (!CPT)
            return false;
          // FIXME: check whether evaluating default arguments can throw.
          // For now, we'll be conservative and assume that they can throw.
          if (!CPT->isNothrow() || CPT->getNumParams() > 0)
            return false;
        }
      }
      return FoundConstructor;
    }
    return false;
  case UTT_HasVirtualDestructor:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If type is a class type with a virtual destructor ([class.dtor])
    //   then the trait is true, else it is false.
    if (CXXRecordDecl *RD = T->getAsCXXRecordDecl())
      if (CXXDestructorDecl *Destructor = Self.LookupDestructor(RD))
        return Destructor->isVirtual();
    return false;

    // These type trait expressions are modeled on the specifications for the
    // Embarcadero C++0x type trait functions:
    //   http://docwiki.embarcadero.com/RADStudio/XE/en/Type_Trait_Functions_(C%2B%2B0x)_Index
  case UTT_IsCompleteType:
    // http://docwiki.embarcadero.com/RADStudio/XE/en/Is_complete_type_(typename_T_):
    //   Returns True if and only if T is a complete type at the point of the
    //   function call.
    return !T->isIncompleteType();
  case UTT_HasUniqueObjectRepresentations:
    return C.hasUniqueObjectRepresentations(T);
  case UTT_IsTriviallyRelocatable:
    return IsTriviallyRelocatableType(Self, T);
  case UTT_IsBitwiseCloneable:
    return T.isBitwiseCloneableType(C);
  case UTT_IsCppTriviallyRelocatable:
    return Self.IsCXXTriviallyRelocatableType(T);
  case UTT_IsReplaceable:
    return Self.IsCXXReplaceableType(T);
  case UTT_CanPassInRegs:
    if (CXXRecordDecl *RD = T->getAsCXXRecordDecl(); RD && !T.hasQualifiers())
      return RD->canPassInRegisters();
    Self.Diag(KeyLoc, diag::err_builtin_pass_in_regs_non_class) << T;
    return false;
  case UTT_IsTriviallyEqualityComparable:
    return isTriviallyEqualityComparableType(Self, T, KeyLoc);
  case UTT_IsImplicitLifetime: {
    DiagnoseVLAInCXXTypeTrait(Self, TInfo,
                              tok::kw___builtin_is_implicit_lifetime);
    DiagnoseAtomicInCXXTypeTrait(Self, TInfo,
                                 tok::kw___builtin_is_implicit_lifetime);

    // [basic.types.general] p9
    // Scalar types, implicit-lifetime class types ([class.prop]),
    // array types, and cv-qualified versions of these types
    // are collectively called implicit-lifetime types.
    QualType UnqualT = T->getCanonicalTypeUnqualified();
    if (UnqualT->isScalarType())
      return true;
    if (UnqualT->isArrayType() || UnqualT->isVectorType())
      return true;
    const CXXRecordDecl *RD = UnqualT->getAsCXXRecordDecl();
    if (!RD)
      return false;

    // [class.prop] p9
    // A class S is an implicit-lifetime class if
    //   - it is an aggregate whose destructor is not user-provided or
    //   - it has at least one trivial eligible constructor and a trivial,
    //     non-deleted destructor.
    const CXXDestructorDecl *Dtor = RD->getDestructor();
    if (UnqualT->isAggregateType())
      if (Dtor && !Dtor->isUserProvided())
        return true;
    if (RD->hasTrivialDestructor() && (!Dtor || !Dtor->isDeleted()))
      if (RD->hasTrivialDefaultConstructor() ||
          RD->hasTrivialCopyConstructor() || RD->hasTrivialMoveConstructor())
        return true;
    return false;
  }
  case UTT_IsIntangibleType:
    assert(Self.getLangOpts().HLSL && "intangible types are HLSL-only feature");
    if (!T->isVoidType() && !T->isIncompleteArrayType())
      if (Self.RequireCompleteType(TInfo->getTypeLoc().getBeginLoc(), T,
                                   diag::err_incomplete_type))
        return false;
    if (DiagnoseVLAInCXXTypeTrait(Self, TInfo,
                                  tok::kw___builtin_hlsl_is_intangible))
      return false;
    return T->isHLSLIntangibleType();

  case UTT_IsTypedResourceElementCompatible:
    assert(Self.getLangOpts().HLSL &&
           "typed resource element compatible types are an HLSL-only feature");
    if (T->isIncompleteType())
      return false;

    return Self.HLSL().IsTypedResourceElementCompatible(T);
  }
}

static bool EvaluateBinaryTypeTrait(Sema &Self, TypeTrait BTT,
                                    const TypeSourceInfo *Lhs,
                                    const TypeSourceInfo *Rhs,
                                    SourceLocation KeyLoc);

static ExprResult CheckConvertibilityForTypeTraits(
    Sema &Self, const TypeSourceInfo *Lhs, const TypeSourceInfo *Rhs,
    SourceLocation KeyLoc, llvm::BumpPtrAllocator &OpaqueExprAllocator) {

  QualType LhsT = Lhs->getType();
  QualType RhsT = Rhs->getType();

  // C++0x [meta.rel]p4:
  //   Given the following function prototype:
  //
  //     template <class T>
  //       typename add_rvalue_reference<T>::type create();
  //
  //   the predicate condition for a template specialization
  //   is_convertible<From, To> shall be satisfied if and only if
  //   the return expression in the following code would be
  //   well-formed, including any implicit conversions to the return
  //   type of the function:
  //
  //     To test() {
  //       return create<From>();
  //     }
  //
  //   Access checking is performed as if in a context unrelated to To and
  //   From. Only the validity of the immediate context of the expression
  //   of the return-statement (including conversions to the return type)
  //   is considered.
  //
  // We model the initialization as a copy-initialization of a temporary
  // of the appropriate type, which for this expression is identical to the
  // return statement (since NRVO doesn't apply).

  // Functions aren't allowed to return function or array types.
  if (RhsT->isFunctionType() || RhsT->isArrayType())
    return ExprError();

  // A function definition requires a complete, non-abstract return type.
  if (!Self.isCompleteType(Rhs->getTypeLoc().getBeginLoc(), RhsT) ||
      Self.isAbstractType(Rhs->getTypeLoc().getBeginLoc(), RhsT))
    return ExprError();

  // Compute the result of add_rvalue_reference.
  if (LhsT->isObjectType() || LhsT->isFunctionType())
    LhsT = Self.Context.getRValueReferenceType(LhsT);

  // Build a fake source and destination for initialization.
  InitializedEntity To(InitializedEntity::InitializeTemporary(RhsT));
  Expr *From = new (OpaqueExprAllocator.Allocate<OpaqueValueExpr>())
      OpaqueValueExpr(KeyLoc, LhsT.getNonLValueExprType(Self.Context),
                      Expr::getValueKindForType(LhsT));
  InitializationKind Kind =
      InitializationKind::CreateCopy(KeyLoc, SourceLocation());

  // Perform the initialization in an unevaluated context within a SFINAE
  // trap at translation unit scope.
  EnterExpressionEvaluationContext Unevaluated(
      Self, Sema::ExpressionEvaluationContext::Unevaluated);
  Sema::SFINAETrap SFINAE(Self, /*ForValidityCheck=*/true);
  Sema::ContextRAII TUContext(Self, Self.Context.getTranslationUnitDecl());
  InitializationSequence Init(Self, To, Kind, From);
  if (Init.Failed())
    return ExprError();

  ExprResult Result = Init.Perform(Self, To, Kind, From);
  if (Result.isInvalid() || SFINAE.hasErrorOccurred())
    return ExprError();

  return Result;
}

static APValue EvaluateSizeTTypeTrait(Sema &S, TypeTrait Kind,
                                      SourceLocation KWLoc,
                                      ArrayRef<TypeSourceInfo *> Args,
                                      SourceLocation RParenLoc,
                                      bool IsDependent) {
  if (IsDependent)
    return APValue();

  switch (Kind) {
  case TypeTrait::UTT_StructuredBindingSize: {
    QualType T = Args[0]->getType();
    SourceRange ArgRange = Args[0]->getTypeLoc().getSourceRange();
    UnsignedOrNone Size =
        S.GetDecompositionElementCount(T, ArgRange.getBegin());
    if (!Size) {
      S.Diag(KWLoc, diag::err_arg_is_not_destructurable) << T << ArgRange;
      return APValue();
    }
    return APValue(
        S.getASTContext().MakeIntValue(*Size, S.getASTContext().getSizeType()));
    break;
  }
  default:
    llvm_unreachable("Not a SizeT type trait");
  }
}

static bool EvaluateBooleanTypeTrait(Sema &S, TypeTrait Kind,
                                     SourceLocation KWLoc,
                                     ArrayRef<TypeSourceInfo *> Args,
                                     SourceLocation RParenLoc,
                                     bool IsDependent) {
  if (IsDependent)
    return false;

  if (Kind <= UTT_Last)
    return EvaluateUnaryTypeTrait(S, Kind, KWLoc, Args[0]);

  // Evaluate ReferenceBindsToTemporary and ReferenceConstructsFromTemporary
  // alongside the IsConstructible traits to avoid duplication.
  if (Kind <= BTT_Last && Kind != BTT_ReferenceBindsToTemporary &&
      Kind != BTT_ReferenceConstructsFromTemporary &&
      Kind != BTT_ReferenceConvertsFromTemporary)
    return EvaluateBinaryTypeTrait(S, Kind, Args[0], Args[1], RParenLoc);

  switch (Kind) {
  case clang::BTT_ReferenceBindsToTemporary:
  case clang::BTT_ReferenceConstructsFromTemporary:
  case clang::BTT_ReferenceConvertsFromTemporary:
  case clang::TT_IsConstructible:
  case clang::TT_IsNothrowConstructible:
  case clang::TT_IsTriviallyConstructible: {
    // C++11 [meta.unary.prop]:
    //   is_trivially_constructible is defined as:
    //
    //     is_constructible<T, Args...>::value is true and the variable
    //     definition for is_constructible, as defined below, is known to call
    //     no operation that is not trivial.
    //
    //   The predicate condition for a template specialization
    //   is_constructible<T, Args...> shall be satisfied if and only if the
    //   following variable definition would be well-formed for some invented
    //   variable t:
    //
    //     T t(create<Args>()...);
    assert(!Args.empty());

    // Precondition: T and all types in the parameter pack Args shall be
    // complete types, (possibly cv-qualified) void, or arrays of
    // unknown bound.
    for (const auto *TSI : Args) {
      QualType ArgTy = TSI->getType();
      if (ArgTy->isVoidType() || ArgTy->isIncompleteArrayType())
        continue;

      if (S.RequireCompleteType(
              KWLoc, ArgTy, diag::err_incomplete_type_used_in_type_trait_expr))
        return false;
    }

    // Make sure the first argument is not incomplete nor a function type.
    QualType T = Args[0]->getType();
    if (T->isIncompleteType() || T->isFunctionType())
      return false;

    // Make sure the first argument is not an abstract type.
    CXXRecordDecl *RD = T->getAsCXXRecordDecl();
    if (RD && RD->isAbstract())
      return false;

    llvm::BumpPtrAllocator OpaqueExprAllocator;
    SmallVector<Expr *, 2> ArgExprs;
    ArgExprs.reserve(Args.size() - 1);
    for (unsigned I = 1, N = Args.size(); I != N; ++I) {
      QualType ArgTy = Args[I]->getType();
      if (ArgTy->isObjectType() || ArgTy->isFunctionType())
        ArgTy = S.Context.getRValueReferenceType(ArgTy);
      ArgExprs.push_back(
          new (OpaqueExprAllocator.Allocate<OpaqueValueExpr>())
              OpaqueValueExpr(Args[I]->getTypeLoc().getBeginLoc(),
                              ArgTy.getNonLValueExprType(S.Context),
                              Expr::getValueKindForType(ArgTy)));
    }

    // Perform the initialization in an unevaluated context within a SFINAE
    // trap at translation unit scope.
    EnterExpressionEvaluationContext Unevaluated(
        S, Sema::ExpressionEvaluationContext::Unevaluated);
    Sema::SFINAETrap SFINAE(S, /*ForValidityCheck=*/true);
    Sema::ContextRAII TUContext(S, S.Context.getTranslationUnitDecl());
    InitializedEntity To(
        InitializedEntity::InitializeTemporary(S.Context, Args[0]));
    InitializationKind InitKind(
        Kind == clang::BTT_ReferenceConvertsFromTemporary
            ? InitializationKind::CreateCopy(KWLoc, KWLoc)
            : InitializationKind::CreateDirect(KWLoc, KWLoc, RParenLoc));
    InitializationSequence Init(S, To, InitKind, ArgExprs);
    if (Init.Failed())
      return false;

    ExprResult Result = Init.Perform(S, To, InitKind, ArgExprs);
    if (Result.isInvalid() || SFINAE.hasErrorOccurred())
      return false;

    if (Kind == clang::TT_IsConstructible)
      return true;

    if (Kind == clang::BTT_ReferenceBindsToTemporary ||
        Kind == clang::BTT_ReferenceConstructsFromTemporary ||
        Kind == clang::BTT_ReferenceConvertsFromTemporary) {
      if (!T->isReferenceType())
        return false;

      if (!Init.isDirectReferenceBinding())
        return true;

      if (Kind == clang::BTT_ReferenceBindsToTemporary)
        return false;

      QualType U = Args[1]->getType();
      if (U->isReferenceType())
        return false;

      TypeSourceInfo *TPtr = S.Context.CreateTypeSourceInfo(
          S.Context.getPointerType(T.getNonReferenceType()));
      TypeSourceInfo *UPtr = S.Context.CreateTypeSourceInfo(
          S.Context.getPointerType(U.getNonReferenceType()));
      return !CheckConvertibilityForTypeTraits(S, UPtr, TPtr, RParenLoc,
                                               OpaqueExprAllocator)
                  .isInvalid();
    }

    if (Kind == clang::TT_IsNothrowConstructible)
      return S.canThrow(Result.get()) == CT_Cannot;

    if (Kind == clang::TT_IsTriviallyConstructible) {
      // Under Objective-C ARC and Weak, if the destination has non-trivial
      // Objective-C lifetime, this is a non-trivial construction.
      if (T.getNonReferenceType().hasNonTrivialObjCLifetime())
        return false;

      // The initialization succeeded; now make sure there are no non-trivial
      // calls.
      return !Result.get()->hasNonTrivialCall(S.Context);
    }

    llvm_unreachable("unhandled type trait");
    return false;
  }
  default:
    llvm_unreachable("not a TT");
  }

  return false;
}

namespace {
void DiagnoseBuiltinDeprecation(Sema &S, TypeTrait Kind, SourceLocation KWLoc) {
  TypeTrait Replacement;
  switch (Kind) {
  case UTT_HasNothrowAssign:
  case UTT_HasNothrowMoveAssign:
    Replacement = BTT_IsNothrowAssignable;
    break;
  case UTT_HasNothrowCopy:
  case UTT_HasNothrowConstructor:
    Replacement = TT_IsNothrowConstructible;
    break;
  case UTT_HasTrivialAssign:
  case UTT_HasTrivialMoveAssign:
    Replacement = BTT_IsTriviallyAssignable;
    break;
  case UTT_HasTrivialCopy:
    Replacement = UTT_IsTriviallyCopyable;
    break;
  case UTT_HasTrivialDefaultConstructor:
  case UTT_HasTrivialMoveConstructor:
    Replacement = TT_IsTriviallyConstructible;
    break;
  case UTT_HasTrivialDestructor:
    Replacement = UTT_IsTriviallyDestructible;
    break;
  case UTT_IsTriviallyRelocatable:
    Replacement = clang::UTT_IsCppTriviallyRelocatable;
    break;
  default:
    return;
  }
  S.Diag(KWLoc, diag::warn_deprecated_builtin)
      << getTraitSpelling(Kind) << getTraitSpelling(Replacement);
}
} // namespace

bool Sema::CheckTypeTraitArity(unsigned Arity, SourceLocation Loc, size_t N) {
  if (Arity && N != Arity) {
    Diag(Loc, diag::err_type_trait_arity)
        << Arity << 0 << (Arity > 1) << (int)N << SourceRange(Loc);
    return false;
  }

  if (!Arity && N == 0) {
    Diag(Loc, diag::err_type_trait_arity)
        << 1 << 1 << 1 << (int)N << SourceRange(Loc);
    return false;
  }
  return true;
}

enum class TypeTraitReturnType {
  Bool,
  SizeT,
};

static TypeTraitReturnType GetReturnType(TypeTrait Kind) {
  if (Kind == TypeTrait::UTT_StructuredBindingSize)
    return TypeTraitReturnType::SizeT;
  return TypeTraitReturnType::Bool;
}

ExprResult Sema::BuildTypeTrait(TypeTrait Kind, SourceLocation KWLoc,
                                ArrayRef<TypeSourceInfo *> Args,
                                SourceLocation RParenLoc) {
  if (!CheckTypeTraitArity(getTypeTraitArity(Kind), KWLoc, Args.size()))
    return ExprError();

  if (Kind <= UTT_Last && !CheckUnaryTypeTraitTypeCompleteness(
                              *this, Kind, KWLoc, Args[0]->getType()))
    return ExprError();

  DiagnoseBuiltinDeprecation(*this, Kind, KWLoc);

  bool Dependent = false;
  for (unsigned I = 0, N = Args.size(); I != N; ++I) {
    if (Args[I]->getType()->isDependentType()) {
      Dependent = true;
      break;
    }
  }

  switch (GetReturnType(Kind)) {
  case TypeTraitReturnType::Bool: {
    bool Result = EvaluateBooleanTypeTrait(*this, Kind, KWLoc, Args, RParenLoc,
                                           Dependent);
    return TypeTraitExpr::Create(Context, Context.getLogicalOperationType(),
                                 KWLoc, Kind, Args, RParenLoc, Result);
  }
  case TypeTraitReturnType::SizeT: {
    APValue Result =
        EvaluateSizeTTypeTrait(*this, Kind, KWLoc, Args, RParenLoc, Dependent);
    return TypeTraitExpr::Create(Context, Context.getSizeType(), KWLoc, Kind,
                                 Args, RParenLoc, Result);
  }
  }
  llvm_unreachable("unhandled type trait return type");
}

ExprResult Sema::ActOnTypeTrait(TypeTrait Kind, SourceLocation KWLoc,
                                ArrayRef<ParsedType> Args,
                                SourceLocation RParenLoc) {
  SmallVector<TypeSourceInfo *, 4> ConvertedArgs;
  ConvertedArgs.reserve(Args.size());

  for (unsigned I = 0, N = Args.size(); I != N; ++I) {
    TypeSourceInfo *TInfo;
    QualType T = GetTypeFromParser(Args[I], &TInfo);
    if (!TInfo)
      TInfo = Context.getTrivialTypeSourceInfo(T, KWLoc);

    ConvertedArgs.push_back(TInfo);
  }

  return BuildTypeTrait(Kind, KWLoc, ConvertedArgs, RParenLoc);
}

static bool EvaluateBinaryTypeTrait(Sema &Self, TypeTrait BTT,
                                    const TypeSourceInfo *Lhs,
                                    const TypeSourceInfo *Rhs,
                                    SourceLocation KeyLoc) {
  QualType LhsT = Lhs->getType();
  QualType RhsT = Rhs->getType();

  assert(!LhsT->isDependentType() && !RhsT->isDependentType() &&
         "Cannot evaluate traits of dependent types");

  switch (BTT) {
  case BTT_IsBaseOf: {
    // C++0x [meta.rel]p2
    // Base is a base class of Derived without regard to cv-qualifiers or
    // Base and Derived are not unions and name the same class type without
    // regard to cv-qualifiers.

    const RecordType *lhsRecord = LhsT->getAs<RecordType>();
    const RecordType *rhsRecord = RhsT->getAs<RecordType>();
    if (!rhsRecord || !lhsRecord) {
      const ObjCObjectType *LHSObjTy = LhsT->getAs<ObjCObjectType>();
      const ObjCObjectType *RHSObjTy = RhsT->getAs<ObjCObjectType>();
      if (!LHSObjTy || !RHSObjTy)
        return false;

      ObjCInterfaceDecl *BaseInterface = LHSObjTy->getInterface();
      ObjCInterfaceDecl *DerivedInterface = RHSObjTy->getInterface();
      if (!BaseInterface || !DerivedInterface)
        return false;

      if (Self.RequireCompleteType(
              Rhs->getTypeLoc().getBeginLoc(), RhsT,
              diag::err_incomplete_type_used_in_type_trait_expr))
        return false;

      return BaseInterface->isSuperClassOf(DerivedInterface);
    }

    assert(Self.Context.hasSameUnqualifiedType(LhsT, RhsT) ==
           (lhsRecord == rhsRecord));

    // Unions are never base classes, and never have base classes.
    // It doesn't matter if they are complete or not. See PR#41843
    if (lhsRecord && lhsRecord->getDecl()->isUnion())
      return false;
    if (rhsRecord && rhsRecord->getDecl()->isUnion())
      return false;

    if (lhsRecord == rhsRecord)
      return true;

    // C++0x [meta.rel]p2:
    //   If Base and Derived are class types and are different types
    //   (ignoring possible cv-qualifiers) then Derived shall be a
    //   complete type.
    if (Self.RequireCompleteType(
            Rhs->getTypeLoc().getBeginLoc(), RhsT,
            diag::err_incomplete_type_used_in_type_trait_expr))
      return false;

    return cast<CXXRecordDecl>(rhsRecord->getDecl())
        ->isDerivedFrom(cast<CXXRecordDecl>(lhsRecord->getDecl()));
  }
  case BTT_IsVirtualBaseOf: {
    const RecordType *BaseRecord = LhsT->getAs<RecordType>();
    const RecordType *DerivedRecord = RhsT->getAs<RecordType>();

    if (!BaseRecord || !DerivedRecord) {
      DiagnoseVLAInCXXTypeTrait(Self, Lhs,
                                tok::kw___builtin_is_virtual_base_of);
      DiagnoseVLAInCXXTypeTrait(Self, Rhs,
                                tok::kw___builtin_is_virtual_base_of);
      return false;
    }

    if (BaseRecord->isUnionType() || DerivedRecord->isUnionType())
      return false;

    if (!BaseRecord->isStructureOrClassType() ||
        !DerivedRecord->isStructureOrClassType())
      return false;

    if (Self.RequireCompleteType(Rhs->getTypeLoc().getBeginLoc(), RhsT,
                                 diag::err_incomplete_type))
      return false;

    return cast<CXXRecordDecl>(DerivedRecord->getDecl())
        ->isVirtuallyDerivedFrom(cast<CXXRecordDecl>(BaseRecord->getDecl()));
  }
  case BTT_IsSame:
    return Self.Context.hasSameType(LhsT, RhsT);
  case BTT_TypeCompatible: {
    // GCC ignores cv-qualifiers on arrays for this builtin.
    Qualifiers LhsQuals, RhsQuals;
    QualType Lhs = Self.getASTContext().getUnqualifiedArrayType(LhsT, LhsQuals);
    QualType Rhs = Self.getASTContext().getUnqualifiedArrayType(RhsT, RhsQuals);
    return Self.Context.typesAreCompatible(Lhs, Rhs);
  }
  case BTT_IsConvertible:
  case BTT_IsConvertibleTo:
  case BTT_IsNothrowConvertible: {
    if (RhsT->isVoidType())
      return LhsT->isVoidType();
    llvm::BumpPtrAllocator OpaqueExprAllocator;
    ExprResult Result = CheckConvertibilityForTypeTraits(Self, Lhs, Rhs, KeyLoc,
                                                         OpaqueExprAllocator);
    if (Result.isInvalid())
      return false;

    if (BTT != BTT_IsNothrowConvertible)
      return true;

    return Self.canThrow(Result.get()) == CT_Cannot;
  }

  case BTT_IsAssignable:
  case BTT_IsNothrowAssignable:
  case BTT_IsTriviallyAssignable: {
    // C++11 [meta.unary.prop]p3:
    //   is_trivially_assignable is defined as:
    //     is_assignable<T, U>::value is true and the assignment, as defined by
    //     is_assignable, is known to call no operation that is not trivial
    //
    //   is_assignable is defined as:
    //     The expression declval<T>() = declval<U>() is well-formed when
    //     treated as an unevaluated operand (Clause 5).
    //
    //   For both, T and U shall be complete types, (possibly cv-qualified)
    //   void, or arrays of unknown bound.
    if (!LhsT->isVoidType() && !LhsT->isIncompleteArrayType() &&
        Self.RequireCompleteType(
            Lhs->getTypeLoc().getBeginLoc(), LhsT,
            diag::err_incomplete_type_used_in_type_trait_expr))
      return false;
    if (!RhsT->isVoidType() && !RhsT->isIncompleteArrayType() &&
        Self.RequireCompleteType(
            Rhs->getTypeLoc().getBeginLoc(), RhsT,
            diag::err_incomplete_type_used_in_type_trait_expr))
      return false;

    // cv void is never assignable.
    if (LhsT->isVoidType() || RhsT->isVoidType())
      return false;

    // Build expressions that emulate the effect of declval<T>() and
    // declval<U>().
    if (LhsT->isObjectType() || LhsT->isFunctionType())
      LhsT = Self.Context.getRValueReferenceType(LhsT);
    if (RhsT->isObjectType() || RhsT->isFunctionType())
      RhsT = Self.Context.getRValueReferenceType(RhsT);
    OpaqueValueExpr Lhs(KeyLoc, LhsT.getNonLValueExprType(Self.Context),
                        Expr::getValueKindForType(LhsT));
    OpaqueValueExpr Rhs(KeyLoc, RhsT.getNonLValueExprType(Self.Context),
                        Expr::getValueKindForType(RhsT));

    // Attempt the assignment in an unevaluated context within a SFINAE
    // trap at translation unit scope.
    EnterExpressionEvaluationContext Unevaluated(
        Self, Sema::ExpressionEvaluationContext::Unevaluated);
    Sema::SFINAETrap SFINAE(Self, /*ForValidityCheck=*/true);
    Sema::ContextRAII TUContext(Self, Self.Context.getTranslationUnitDecl());
    ExprResult Result =
        Self.BuildBinOp(/*S=*/nullptr, KeyLoc, BO_Assign, &Lhs, &Rhs);
    if (Result.isInvalid())
      return false;

    // Treat the assignment as unused for the purpose of -Wdeprecated-volatile.
    Self.CheckUnusedVolatileAssignment(Result.get());

    if (SFINAE.hasErrorOccurred())
      return false;

    if (BTT == BTT_IsAssignable)
      return true;

    if (BTT == BTT_IsNothrowAssignable)
      return Self.canThrow(Result.get()) == CT_Cannot;

    if (BTT == BTT_IsTriviallyAssignable) {
      // Under Objective-C ARC and Weak, if the destination has non-trivial
      // Objective-C lifetime, this is a non-trivial assignment.
      if (LhsT.getNonReferenceType().hasNonTrivialObjCLifetime())
        return false;

      return !Result.get()->hasNonTrivialCall(Self.Context);
    }

    llvm_unreachable("unhandled type trait");
    return false;
  }
  case BTT_IsLayoutCompatible: {
    if (!LhsT->isVoidType() && !LhsT->isIncompleteArrayType())
      Self.RequireCompleteType(Lhs->getTypeLoc().getBeginLoc(), LhsT,
                               diag::err_incomplete_type);
    if (!RhsT->isVoidType() && !RhsT->isIncompleteArrayType())
      Self.RequireCompleteType(Rhs->getTypeLoc().getBeginLoc(), RhsT,
                               diag::err_incomplete_type);

    DiagnoseVLAInCXXTypeTrait(Self, Lhs, tok::kw___is_layout_compatible);
    DiagnoseVLAInCXXTypeTrait(Self, Rhs, tok::kw___is_layout_compatible);

    return Self.IsLayoutCompatible(LhsT, RhsT);
  }
  case BTT_IsPointerInterconvertibleBaseOf: {
    if (LhsT->isStructureOrClassType() && RhsT->isStructureOrClassType() &&
        !Self.getASTContext().hasSameUnqualifiedType(LhsT, RhsT)) {
      Self.RequireCompleteType(Rhs->getTypeLoc().getBeginLoc(), RhsT,
                               diag::err_incomplete_type);
    }

    DiagnoseVLAInCXXTypeTrait(Self, Lhs,
                              tok::kw___is_pointer_interconvertible_base_of);
    DiagnoseVLAInCXXTypeTrait(Self, Rhs,
                              tok::kw___is_pointer_interconvertible_base_of);

    return Self.IsPointerInterconvertibleBaseOf(Lhs, Rhs);
  }
  case BTT_IsDeducible: {
    const auto *TSTToBeDeduced = cast<DeducedTemplateSpecializationType>(LhsT);
    sema::TemplateDeductionInfo Info(KeyLoc);
    return Self.DeduceTemplateArgumentsFromType(
               TSTToBeDeduced->getTemplateName().getAsTemplateDecl(), RhsT,
               Info) == TemplateDeductionResult::Success;
  }
  case BTT_IsScalarizedLayoutCompatible: {
    if (!LhsT->isVoidType() && !LhsT->isIncompleteArrayType() &&
        Self.RequireCompleteType(Lhs->getTypeLoc().getBeginLoc(), LhsT,
                                 diag::err_incomplete_type))
      return true;
    if (!RhsT->isVoidType() && !RhsT->isIncompleteArrayType() &&
        Self.RequireCompleteType(Rhs->getTypeLoc().getBeginLoc(), RhsT,
                                 diag::err_incomplete_type))
      return true;

    DiagnoseVLAInCXXTypeTrait(
        Self, Lhs, tok::kw___builtin_hlsl_is_scalarized_layout_compatible);
    DiagnoseVLAInCXXTypeTrait(
        Self, Rhs, tok::kw___builtin_hlsl_is_scalarized_layout_compatible);

    return Self.HLSL().IsScalarizedLayoutCompatible(LhsT, RhsT);
  }
  default:
    llvm_unreachable("not a BTT");
  }
  llvm_unreachable("Unknown type trait or not implemented");
}

ExprResult Sema::ActOnArrayTypeTrait(ArrayTypeTrait ATT, SourceLocation KWLoc,
                                     ParsedType Ty, Expr *DimExpr,
                                     SourceLocation RParen) {
  TypeSourceInfo *TSInfo;
  QualType T = GetTypeFromParser(Ty, &TSInfo);
  if (!TSInfo)
    TSInfo = Context.getTrivialTypeSourceInfo(T);

  return BuildArrayTypeTrait(ATT, KWLoc, TSInfo, DimExpr, RParen);
}

static uint64_t EvaluateArrayTypeTrait(Sema &Self, ArrayTypeTrait ATT,
                                       QualType T, Expr *DimExpr,
                                       SourceLocation KeyLoc) {
  assert(!T->isDependentType() && "Cannot evaluate traits of dependent type");

  switch (ATT) {
  case ATT_ArrayRank:
    if (T->isArrayType()) {
      unsigned Dim = 0;
      while (const ArrayType *AT = Self.Context.getAsArrayType(T)) {
        ++Dim;
        T = AT->getElementType();
      }
      return Dim;
    }
    return 0;

  case ATT_ArrayExtent: {
    llvm::APSInt Value;
    uint64_t Dim;
    if (Self.VerifyIntegerConstantExpression(
                DimExpr, &Value, diag::err_dimension_expr_not_constant_integer)
            .isInvalid())
      return 0;
    if (Value.isSigned() && Value.isNegative()) {
      Self.Diag(KeyLoc, diag::err_dimension_expr_not_constant_integer)
          << DimExpr->getSourceRange();
      return 0;
    }
    Dim = Value.getLimitedValue();

    if (T->isArrayType()) {
      unsigned D = 0;
      bool Matched = false;
      while (const ArrayType *AT = Self.Context.getAsArrayType(T)) {
        if (Dim == D) {
          Matched = true;
          break;
        }
        ++D;
        T = AT->getElementType();
      }

      if (Matched && T->isArrayType()) {
        if (const ConstantArrayType *CAT =
                Self.Context.getAsConstantArrayType(T))
          return CAT->getLimitedSize();
      }
    }
    return 0;
  }
  }
  llvm_unreachable("Unknown type trait or not implemented");
}

ExprResult Sema::BuildArrayTypeTrait(ArrayTypeTrait ATT, SourceLocation KWLoc,
                                     TypeSourceInfo *TSInfo, Expr *DimExpr,
                                     SourceLocation RParen) {
  QualType T = TSInfo->getType();

  // FIXME: This should likely be tracked as an APInt to remove any host
  // assumptions about the width of size_t on the target.
  uint64_t Value = 0;
  if (!T->isDependentType())
    Value = EvaluateArrayTypeTrait(*this, ATT, T, DimExpr, KWLoc);

  // While the specification for these traits from the Embarcadero C++
  // compiler's documentation says the return type is 'unsigned int', Clang
  // returns 'size_t'. On Windows, the primary platform for the Embarcadero
  // compiler, there is no difference. On several other platforms this is an
  // important distinction.
  return new (Context) ArrayTypeTraitExpr(KWLoc, ATT, TSInfo, Value, DimExpr,
                                          RParen, Context.getSizeType());
}

ExprResult Sema::ActOnExpressionTrait(ExpressionTrait ET, SourceLocation KWLoc,
                                      Expr *Queried, SourceLocation RParen) {
  // If error parsing the expression, ignore.
  if (!Queried)
    return ExprError();

  ExprResult Result = BuildExpressionTrait(ET, KWLoc, Queried, RParen);

  return Result;
}

static bool EvaluateExpressionTrait(ExpressionTrait ET, Expr *E) {
  switch (ET) {
  case ET_IsLValueExpr:
    return E->isLValue();
  case ET_IsRValueExpr:
    return E->isPRValue();
  }
  llvm_unreachable("Expression trait not covered by switch");
}

ExprResult Sema::BuildExpressionTrait(ExpressionTrait ET, SourceLocation KWLoc,
                                      Expr *Queried, SourceLocation RParen) {
  if (Queried->isTypeDependent()) {
    // Delay type-checking for type-dependent expressions.
  } else if (Queried->hasPlaceholderType()) {
    ExprResult PE = CheckPlaceholderExpr(Queried);
    if (PE.isInvalid())
      return ExprError();
    return BuildExpressionTrait(ET, KWLoc, PE.get(), RParen);
  }

  bool Value = EvaluateExpressionTrait(ET, Queried);

  return new (Context)
      ExpressionTraitExpr(KWLoc, ET, Queried, Value, RParen, Context.BoolTy);
}
