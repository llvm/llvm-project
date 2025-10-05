//===-- SemaBoundsSafety.cpp - Bounds Safety specific routines-*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis functions specific to `-fbounds-safety`
/// (Bounds Safety) and also its attributes when used without `-fbounds-safety`
/// (e.g. `counted_by`)
///
//===----------------------------------------------------------------------===//
#include "clang/Lex/Lexer.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Sema.h"

namespace clang {

static CountAttributedType::DynamicCountPointerKind
getCountAttrKind(bool CountInBytes, bool OrNull) {
  if (CountInBytes)
    return OrNull ? CountAttributedType::SizedByOrNull
                  : CountAttributedType::SizedBy;
  return OrNull ? CountAttributedType::CountedByOrNull
                : CountAttributedType::CountedBy;
}

static const RecordDecl *GetEnclosingNamedOrTopAnonRecord(const FieldDecl *FD) {
  const auto *RD = FD->getParent();
  // An unnamed struct is anonymous struct only if it's not instantiated.
  // However, the struct may not be fully processed yet to determine
  // whether it's anonymous or not. In that case, this function treats it as
  // an anonymous struct and tries to find a named parent.
  while (RD && (RD->isAnonymousStructOrUnion() ||
                (!RD->isCompleteDefinition() && RD->getName().empty()))) {
    const auto *Parent = dyn_cast<RecordDecl>(RD->getParent());
    if (!Parent)
      break;
    RD = Parent;
  }
  return RD;
}

enum class CountedByInvalidPointeeTypeKind {
  INCOMPLETE,
  SIZELESS,
  FUNCTION,
  FLEXIBLE_ARRAY_MEMBER,
  VALID,
};

bool Sema::CheckCountedByAttrOnField(FieldDecl *FD, Expr *E, bool CountInBytes,
                                     bool OrNull) {
  // Check the context the attribute is used in

  unsigned Kind = getCountAttrKind(CountInBytes, OrNull);

  if (FD->getParent()->isUnion()) {
    Diag(FD->getBeginLoc(), diag::err_count_attr_in_union)
        << Kind << FD->getSourceRange();
    return true;
  }

  const auto FieldTy = FD->getType();
  if (FieldTy->isArrayType() && (CountInBytes || OrNull)) {
    Diag(FD->getBeginLoc(),
         diag::err_count_attr_not_on_ptr_or_flexible_array_member)
        << Kind << FD->getLocation() << /* suggest counted_by */ 1;
    return true;
  }
  if (!FieldTy->isArrayType() && !FieldTy->isPointerType()) {
    Diag(FD->getBeginLoc(),
         diag::err_count_attr_not_on_ptr_or_flexible_array_member)
        << Kind << FD->getLocation() << /* do not suggest counted_by */ 0;
    return true;
  }

  LangOptions::StrictFlexArraysLevelKind StrictFlexArraysLevel =
      LangOptions::StrictFlexArraysLevelKind::IncompleteOnly;
  if (FieldTy->isArrayType() &&
      !Decl::isFlexibleArrayMemberLike(getASTContext(), FD, FieldTy,
                                       StrictFlexArraysLevel, true)) {
    Diag(FD->getBeginLoc(),
         diag::err_counted_by_attr_on_array_not_flexible_array_member)
        << Kind << FD->getLocation();
    return true;
  }

  CountedByInvalidPointeeTypeKind InvalidTypeKind =
      CountedByInvalidPointeeTypeKind::VALID;
  QualType PointeeTy;
  int SelectPtrOrArr = 0;
  if (FieldTy->isPointerType()) {
    PointeeTy = FieldTy->getPointeeType();
    SelectPtrOrArr = 0;
  } else {
    assert(FieldTy->isArrayType());
    const ArrayType *AT = getASTContext().getAsArrayType(FieldTy);
    PointeeTy = AT->getElementType();
    SelectPtrOrArr = 1;
  }
  // Note: The `Decl::isFlexibleArrayMemberLike` check earlier on means
  // only `PointeeTy->isStructureTypeWithFlexibleArrayMember()` is reachable
  // when `FieldTy->isArrayType()`.
  bool ShouldWarn = false;
  if (PointeeTy->isAlwaysIncompleteType() && !CountInBytes) {
    // In general using `counted_by` or `counted_by_or_null` on
    // pointers where the pointee is an incomplete type are problematic. This is
    // because it isn't possible to compute the pointer's bounds without knowing
    // the pointee type size. At the same time it is common to forward declare
    // types in header files.
    //
    // E.g.:
    //
    // struct Handle;
    // struct Wrapper {
    //   size_t size;
    //   struct Handle* __counted_by(count) handles;
    // }
    //
    // To allow the above code pattern but still prevent the pointee type from
    // being incomplete in places where bounds checks are needed the following
    // scheme is used:
    //
    // * When the pointee type might not always be an incomplete type (i.e.
    // a type that is currently incomplete but might be completed later
    // on in the translation unit) the attribute is allowed by this method
    // but later uses of the FieldDecl are checked that the pointee type
    // is complete see `BoundsSafetyCheckAssignmentToCountAttrPtr`,
    // `BoundsSafetyCheckInitialization`, and
    // `BoundsSafetyCheckUseOfCountAttrPtr`
    //
    // * When the pointee type is always an incomplete type (e.g.
    // `void`) the attribute is disallowed by this method because we know the
    // type can never be completed so there's no reason to allow it.
    InvalidTypeKind = CountedByInvalidPointeeTypeKind::INCOMPLETE;
  } else if (PointeeTy->isSizelessType()) {
    InvalidTypeKind = CountedByInvalidPointeeTypeKind::SIZELESS;
  } else if (PointeeTy->isFunctionType()) {
    InvalidTypeKind = CountedByInvalidPointeeTypeKind::FUNCTION;
  } else if (PointeeTy->isStructureTypeWithFlexibleArrayMember()) {
    if (FieldTy->isArrayType() && !getLangOpts().BoundsSafety) {
      // This is a workaround for the Linux kernel that has already adopted
      // `counted_by` on a FAM where the pointee is a struct with a FAM. This
      // should be an error because computing the bounds of the array cannot be
      // done correctly without manually traversing every struct object in the
      // array at runtime. To allow the code to be built this error is
      // downgraded to a warning.
      ShouldWarn = true;
    }
    InvalidTypeKind = CountedByInvalidPointeeTypeKind::FLEXIBLE_ARRAY_MEMBER;
  }

  if (InvalidTypeKind != CountedByInvalidPointeeTypeKind::VALID) {
    unsigned DiagID = ShouldWarn
                          ? diag::warn_counted_by_attr_elt_type_unknown_size
                          : diag::err_counted_by_attr_pointee_unknown_size;
    Diag(FD->getBeginLoc(), DiagID)
        << SelectPtrOrArr << PointeeTy << (int)InvalidTypeKind
        << (ShouldWarn ? 1 : 0) << Kind << FD->getSourceRange();
    return true;
  }

  // Check the expression

  if (!E->getType()->isIntegerType() || E->getType()->isBooleanType()) {
    Diag(E->getBeginLoc(), diag::err_count_attr_argument_not_integer)
        << Kind << E->getSourceRange();
    return true;
  }

  auto *DRE = dyn_cast<DeclRefExpr>(E);
  if (!DRE) {
    Diag(E->getBeginLoc(),
         diag::err_count_attr_only_support_simple_decl_reference)
        << Kind << E->getSourceRange();
    return true;
  }

  auto *CountDecl = DRE->getDecl();
  FieldDecl *CountFD = dyn_cast<FieldDecl>(CountDecl);
  if (auto *IFD = dyn_cast<IndirectFieldDecl>(CountDecl)) {
    CountFD = IFD->getAnonField();
  }
  if (!CountFD) {
    Diag(E->getBeginLoc(), diag::err_count_attr_must_be_in_structure)
        << CountDecl << Kind << E->getSourceRange();

    Diag(CountDecl->getBeginLoc(),
         diag::note_flexible_array_counted_by_attr_field)
        << CountDecl << CountDecl->getSourceRange();
    return true;
  }

  if (FD->getParent() != CountFD->getParent()) {
    if (CountFD->getParent()->isUnion()) {
      Diag(CountFD->getBeginLoc(), diag::err_count_attr_refer_to_union)
          << Kind << CountFD->getSourceRange();
      return true;
    }
    // Whether CountRD is an anonymous struct is not determined at this
    // point. Thus, an additional diagnostic in case it's not anonymous struct
    // is done later in `Parser::ParseStructDeclaration`.
    auto *RD = GetEnclosingNamedOrTopAnonRecord(FD);
    auto *CountRD = GetEnclosingNamedOrTopAnonRecord(CountFD);

    if (RD != CountRD) {
      Diag(E->getBeginLoc(), diag::err_count_attr_param_not_in_same_struct)
          << CountFD << Kind << FieldTy->isArrayType() << E->getSourceRange();
      Diag(CountFD->getBeginLoc(),
           diag::note_flexible_array_counted_by_attr_field)
          << CountFD << CountFD->getSourceRange();
      return true;
    }
  }
  return false;
}

static void EmitIncompleteCountedByPointeeNotes(Sema &S,
                                                const CountAttributedType *CATy,
                                                NamedDecl *IncompleteTyDecl) {
  assert(IncompleteTyDecl == nullptr || isa<TypeDecl>(IncompleteTyDecl));

  if (IncompleteTyDecl) {
    // Suggest completing the pointee type if its a named typed (i.e.
    // IncompleteTyDecl isn't nullptr). Suggest this first as it is more likely
    // to be the correct fix.
    //
    // Note the `IncompleteTyDecl` type is the underlying type which might not
    // be the same as `CATy->getPointeeType()` which could be a typedef.
    //
    // The diagnostic printed will be at the location of the underlying type but
    // the diagnostic text will print the type of `CATy->getPointeeType()` which
    // could be a typedef name rather than the underlying type. This is ok
    // though because the diagnostic will print the underlying type name too.
    S.Diag(IncompleteTyDecl->getBeginLoc(),
           diag::note_counted_by_consider_completing_pointee_ty)
        << CATy->getPointeeType();
  }

  // Suggest using __sized_by(_or_null) instead of __counted_by(_or_null) as
  // __sized_by(_or_null) doesn't have the complete type restriction.
  //
  // We use the source range of the expression on the CountAttributedType as an
  // approximation for the source range of the attribute. This isn't quite right
  // but isn't easy to fix right now.
  //
  // TODO: Implement logic to find the relevant TypeLoc for the attribute and
  // get the SourceRange from that (#113582).
  //
  // TODO: We should emit a fix-it here.
  SourceRange AttrSrcRange = CATy->getCountExpr()->getSourceRange();
  S.Diag(AttrSrcRange.getBegin(), diag::note_counted_by_consider_using_sized_by)
      << CATy->isOrNull() << AttrSrcRange;
}

static std::tuple<const CountAttributedType *, QualType>
GetCountedByAttrOnIncompletePointee(QualType Ty, NamedDecl **ND) {
  auto *CATy = Ty->getAs<CountAttributedType>();
  // Incomplete pointee type is only a problem for
  // counted_by/counted_by_or_null
  if (!CATy || CATy->isCountInBytes())
    return {};

  auto PointeeTy = CATy->getPointeeType();
  if (PointeeTy.isNull()) {
    // Reachable if `CountAttributedType` wraps an IncompleteArrayType
    return {};
  }

  if (!PointeeTy->isIncompleteType(ND))
    return {};

  return {CATy, PointeeTy};
}

/// Perform Checks for assigning to a `__counted_by` or
/// `__counted_by_or_null` pointer type \param LHSTy where the pointee type
/// is incomplete which is invalid.
///
/// \param S The Sema instance.
/// \param LHSTy The type being assigned to. Checks will only be performed if
///              the type is a `counted_by` or `counted_by_or_null ` pointer.
/// \param RHSExpr The expression being assigned from.
/// \param Action The type assignment being performed
/// \param Loc The SourceLocation to use for error diagnostics
/// \param Assignee The ValueDecl being assigned. This is used to compute
///        the name of the assignee. If the assignee isn't known this can
///        be set to nullptr.
/// \param ShowFullyQualifiedAssigneeName If set to true when using \p
///        Assignee to compute the name of the assignee use the fully
///        qualified name, otherwise use the unqualified name.
///
/// \returns True iff no diagnostic where emitted, false otherwise.
static bool CheckAssignmentToCountAttrPtrWithIncompletePointeeTy(
    Sema &S, QualType LHSTy, Expr *RHSExpr, AssignmentAction Action,
    SourceLocation Loc, const ValueDecl *Assignee,
    bool ShowFullyQualifiedAssigneeName) {
  NamedDecl *IncompleteTyDecl = nullptr;
  auto [CATy, PointeeTy] =
      GetCountedByAttrOnIncompletePointee(LHSTy, &IncompleteTyDecl);
  if (!CATy)
    return true;

  std::string AssigneeStr;
  if (Assignee) {
    if (ShowFullyQualifiedAssigneeName) {
      AssigneeStr = Assignee->getQualifiedNameAsString();
    } else {
      AssigneeStr = Assignee->getNameAsString();
    }
  }

  S.Diag(Loc, diag::err_counted_by_on_incomplete_type_on_assign)
      << static_cast<int>(Action) << AssigneeStr << (AssigneeStr.size() > 0)
      << isa<ImplicitValueInitExpr>(RHSExpr) << LHSTy
      << CATy->getAttributeName(/*WithMacroPrefix=*/true) << PointeeTy
      << CATy->isOrNull() << RHSExpr->getSourceRange();

  EmitIncompleteCountedByPointeeNotes(S, CATy, IncompleteTyDecl);
  return false; // check failed
}

bool Sema::BoundsSafetyCheckAssignmentToCountAttrPtr(
    QualType LHSTy, Expr *RHSExpr, AssignmentAction Action, SourceLocation Loc,
    const ValueDecl *Assignee, bool ShowFullyQualifiedAssigneeName) {
  return CheckAssignmentToCountAttrPtrWithIncompletePointeeTy(
      *this, LHSTy, RHSExpr, Action, Loc, Assignee,
      ShowFullyQualifiedAssigneeName);
}

bool Sema::BoundsSafetyCheckInitialization(const InitializedEntity &Entity,
                                           const InitializationKind &Kind,
                                           AssignmentAction Action,
                                           QualType LHSType, Expr *RHSExpr) {
  auto SL = Kind.getLocation();

  // Note: We don't call `BoundsSafetyCheckAssignmentToCountAttrPtr` here
  // because we need conditionalize what is checked. In downstream
  // Clang `counted_by` is supported on variable definitions and in that
  // implementation an error diagnostic will be emitted on the variable
  // definition if the pointee is an incomplete type. To avoid warning about the
  // same problem twice (once when the variable is defined, once when Sema
  // checks the initializer) we skip checking the initializer if it's a
  // variable.
  if (Action == AssignmentAction::Initializing &&
      Entity.getKind() != InitializedEntity::EK_Variable) {

    if (!CheckAssignmentToCountAttrPtrWithIncompletePointeeTy(
            *this, LHSType, RHSExpr, Action, SL,
            dyn_cast_or_null<ValueDecl>(Entity.getDecl()),
            /*ShowFullQualifiedAssigneeName=*/true)) {
      return false;
    }
  }

  return true;
}

bool Sema::BoundsSafetyCheckUseOfCountAttrPtr(const Expr *E) {
  QualType T = E->getType();
  if (!T->isPointerType())
    return true;

  NamedDecl *IncompleteTyDecl = nullptr;
  auto [CATy, PointeeTy] =
      GetCountedByAttrOnIncompletePointee(T, &IncompleteTyDecl);
  if (!CATy)
    return true;

  // Generate a string for the diagnostic that describes the "use".
  // The string is specialized for direct calls to produce a better
  // diagnostic.
  SmallString<64> UseStr;
  bool IsDirectCall = false;
  if (const auto *CE = dyn_cast<CallExpr>(E->IgnoreParens())) {
    if (const auto *FD = CE->getDirectCallee()) {
      UseStr = FD->getName();
      IsDirectCall = true;
    }
  }

  if (!IsDirectCall) {
    llvm::raw_svector_ostream SS(UseStr);
    E->printPretty(SS, nullptr, getPrintingPolicy());
  }

  Diag(E->getBeginLoc(), diag::err_counted_by_on_incomplete_type_on_use)
      << IsDirectCall << UseStr << T << PointeeTy
      << CATy->getAttributeName(/*WithMacroPrefix=*/true) << CATy->isOrNull()
      << E->getSourceRange();

  EmitIncompleteCountedByPointeeNotes(*this, CATy, IncompleteTyDecl);
  return false;
}

} // namespace clang
