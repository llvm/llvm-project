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

static CountAttributedType::BoundsAttrKind getCountAttrKind(bool CountInBytes,
                                                            bool OrNull) {
  if (CountInBytes)
    return OrNull ? CountAttributedType::SizedByOrNull
                  : CountAttributedType::SizedBy;
  return OrNull ? CountAttributedType::CountedByOrNull
                : CountAttributedType::CountedBy;
}

BoundsAttributedType::BoundsAttrKind
Sema::getBoundsAttrKind(const BoundsAttrFlags &Flags) {
  // Clang does not yet implement __ended_by (see BoundsSafety.md).
  assert(!Flags.IsEndedBy && "ended_by is not implemented");
  return getCountAttrKind(Flags.CountInBytes, Flags.OrNull);
}

Sema::BoundsAttrFlags Sema::BoundsAttrFlags::get(AttributeCommonInfo::Kind K) {
  BoundsAttrFlags Flags;
  switch (K) {
  case ParsedAttr::AT_SizedBy:
    Flags.CountInBytes = true;
    break;
  case ParsedAttr::AT_SizedByOrNull:
    Flags.CountInBytes = true;
    Flags.OrNull = true;
    break;
  case ParsedAttr::AT_CountedBy:
    break;
  case ParsedAttr::AT_CountedByOrNull:
    Flags.OrNull = true;
    break;
  default:
    llvm_unreachable("unexpected bounds attribute kind");
  }
  return Flags;
}

DeclRefExpr *Sema::getCountDeclRef(Expr *E, bool &IsDeref) {
  IsDeref = false;
  if (auto *UO = dyn_cast<UnaryOperator>(E);
      UO && UO->getOpcode() == UO_Deref) {
    E = UO->getSubExpr()->IgnoreImpCasts();
    IsDeref = true;
  }
  return dyn_cast<DeclRefExpr>(E);
}

static const RecordDecl *GetEnclosingNamedOrTopAnonRecord(const FieldDecl *FD) {
  const auto *RD = FD->getParent();
  // An unnamed struct is treated as anonymous struct at this point.
  // A struct may not be fully processed yet to determine
  // whether it's anonymous or not. In that case, this function treats it as
  // an anonymous struct and tries to find a named parent.
  while (RD && (RD->isAnonymousStructOrUnion() || RD->getName().empty())) {
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

/// Check type conflicts under the -fbounds-safety attribute model.
/// Return std::nullopt if validation should continue with the shared checks.
static std::optional<bool> checkBoundsAttrTypeConflictsAndMisc(
    Sema &S, QualType Ty, SourceLocation AttrLoc,
    const Sema::BoundsAttrFlags &Flags, StringRef AttrSpelling,
    bool AllowRedecl, Expr *AttrArg) {
  // The -fbounds-safety model's checks for conflicting attributes, atomic
  // pointers and array shapes are not upstream yet.
  return std::nullopt;
}

bool Sema::ValidateBoundsAttrTypeShape(QualType Ty, SourceLocation AttrLoc,
                                       SourceRange AttrRange,
                                       BoundsAttrFlags &Flags,
                                       StringRef AttrSpelling, bool AllowRedecl,
                                       Expr *AttrArg, bool UpdateFlags) {
  if (getLangOpts().hasBoundsSafetyAttributes())
    if (std::optional<bool> Result = checkBoundsAttrTypeConflictsAndMisc(
            *this, Ty, AttrLoc, Flags, AttrSpelling, AllowRedecl, AttrArg))
      return *Result;

  BoundsAttributedType::BoundsAttrKind Kind = getBoundsAttrKind(Flags);

  // counted_by and sized_by require a pointer or array.
  if (!Ty->isPointerType() && !Ty->isArrayType()) {
    Diag(AttrLoc, diag::err_count_attr_not_on_ptr_or_flexible_array_member)
        << Kind << 0;
    return false;
  }

  // Arrays only support counted_by outside the -fbounds-safety attribute model.
  if (!getLangOpts().hasBoundsSafetyAttributes() && Ty->isArrayType() &&
      (Flags.CountInBytes || Flags.OrNull)) {
    Diag(AttrLoc, diag::err_count_attr_not_on_ptr_or_flexible_array_member)
        << Kind << /*suggest counted_by*/ 1;
    return false;
  }

  // Check the pointee or element type.
  QualType PointeeTy;
  int SelectPtrOrArr;
  if (Ty->isPointerType()) {
    PointeeTy = Ty->getPointeeType();
    SelectPtrOrArr = 0;
  } else {
    const ArrayType *AT = getASTContext().getAsArrayType(Ty);
    PointeeTy = AT->getElementType();
    SelectPtrOrArr = 1;
  }

  auto InvalidTypeKind = CountedByInvalidPointeeTypeKind::VALID;
  bool ShouldWarn = false;
  if (!Flags.CountInBytes && PointeeTy->isAlwaysIncompleteType()) {
    // GNU void pointer arithmetic treats each element as one byte, so allow
    // counted_by on void pointers outside the -fbounds-safety attribute model.
    if (PointeeTy->isVoidType() && !getLangOpts().hasBoundsSafetyAttributes()) {
      Diag(AttrLoc, diag::ext_gnu_counted_by_void_ptr) << Kind;
      Diag(AttrLoc, diag::note_gnu_counted_by_void_ptr_use_sized_by) << Kind;
      if (UpdateFlags)
        Flags.CountInBytes = true;
      return true;
    }
    InvalidTypeKind = CountedByInvalidPointeeTypeKind::INCOMPLETE;
  } else if (PointeeTy->isSizelessType()) {
    InvalidTypeKind = CountedByInvalidPointeeTypeKind::SIZELESS;
  } else if (PointeeTy->isFunctionType()) {
    InvalidTypeKind = CountedByInvalidPointeeTypeKind::FUNCTION;
  } else if (!Flags.CountInBytes &&
             PointeeTy->isStructureTypeWithFlexibleArrayMember()) {
    if (Ty->isArrayType() && !getLangOpts().BoundsSafety) {
      // Warn for compatibility with Linux kernel code using counted_by on a FAM
      // of structs with FAMs. Computing these bounds requires traversing the
      // elements at runtime.
      ShouldWarn = true;
    }
    InvalidTypeKind = CountedByInvalidPointeeTypeKind::FLEXIBLE_ARRAY_MEMBER;
  }

  if (InvalidTypeKind != CountedByInvalidPointeeTypeKind::VALID) {
    unsigned DiagID = ShouldWarn
                          ? diag::warn_counted_by_attr_elt_type_unknown_size
                          : diag::err_counted_by_attr_pointee_unknown_size;
    Diag(AttrLoc, DiagID) << SelectPtrOrArr << PointeeTy << (int)InvalidTypeKind
                          << (ShouldWarn ? 1 : 0) << Kind << AttrRange;
    return false;
  }

  return true;
}

bool Sema::ValidateBoundsAttrTypeForTypePosition(
    QualType Ty, AttributeCommonInfo::Kind AttrKind, SourceLocation AttrLoc,
    SourceRange AttrRange, unsigned PointerNestLevel, BoundsAttrFlags &Flags,
    bool IsParameter) {
  Flags = BoundsAttrFlags::get(AttrKind);

  // An array parameter adjusts to a pointer, which is what the attribute
  // describes. Whether the array may have a size is checked on the parameter.
  if (IsParameter && Ty->isArrayType())
    Ty = getASTContext().getArrayDecayedType(Ty);

  // Preserve the counted_by kind for the GNU void pointer extension to match
  // the field path. The count is still interpreted as a byte size.
  if (!ValidateBoundsAttrTypeShape(Ty, AttrLoc, AttrRange, Flags,
                                   /*AttrSpelling=*/{}, /*AllowRedecl=*/false,
                                   /*AttrArg=*/nullptr, /*UpdateFlags=*/false))
    return false;

  // Currently, only attributes at the outermost level of the declared type
  // are supported. The caller passes zero for the pointer that a parameter
  // points to, as for an out parameter: `int *__counted_by(*len) *buf`.
  if (PointerNestLevel > 0) {
    Diag(AttrLoc, diag::err_counted_by_on_nested_pointer)
        << getBoundsAttrKind(Flags);
    return false;
  }

  return true;
}

bool Sema::CheckCountedByAttrOnField(FieldDecl *FD, Expr *E, bool CountInBytes,
                                     bool OrNull) {
  // Check the context the attribute is used in

  unsigned Kind = getCountAttrKind(CountInBytes, OrNull);

  if (FD->getParent()->isUnion()) {
    Diag(FD->getBeginLoc(), diag::err_count_attr_in_union)
        << Kind << FD->getSourceRange();
    return true;
  }

  const QualType FieldTy = FD->getType();
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

  // Validate the expression type
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

  // Validate count field references
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

bool Sema::CheckCountedByAttrOnParam(QualType ParamTy, Expr *E,
                                     bool CountInBytes, bool OrNull) {
  // An invalid count was already diagnosed.
  if (E->containsErrors())
    return true;

  // A second count would replace the first, or hide it.
  if (ParamTy->getAs<CountAttributedType>()) {
    Diag(E->getBeginLoc(), diag::err_count_attr_more_than_one);
    return true;
  }

  // An array parameter adjusts to a pointer, which the count then describes.
  // An array with an explicit size is not supported: in the -fbounds-safety
  // programming model, its size becomes that pointer's count.
  unsigned Kind = getCountAttrKind(CountInBytes, OrNull);
  if (ParamTy->isArrayType() && !ParamTy->isIncompleteArrayType()) {
    Diag(E->getBeginLoc(), diag::err_count_attr_on_sized_array_param) << Kind;
    return true;
  }

  if (!E->getType()->isIntegerType() || E->getType()->isBooleanType()) {
    Diag(E->getBeginLoc(), diag::err_count_attr_argument_not_integer)
        << Kind << E->getSourceRange();
    return true;
  }

  bool IsDeref;
  auto *DRE = getCountDeclRef(E, IsDeref);
  if (!DRE) {
    Diag(E->getBeginLoc(),
         diag::err_count_attr_only_support_simple_decl_reference)
        << Kind << E->getSourceRange();
    return true;
  }

  // The parameters of an enclosing function declarator are in scope too.
  if (!isa<ParmVarDecl>(DRE->getDecl())) {
    Diag(E->getBeginLoc(), diag::err_count_attr_refer_to_non_param)
        << E->getSourceRange();
    return true;
  }
  return false;
}

void Sema::AdjustCountedArrayParamType(ParmVarDecl *PVD,
                                       const CountAttributedType *CATy) {
  if (!CATy->desugar()->isArrayType())
    return;
  // The parameter's type is already adjusted, with the qualifiers written
  // outside the count moved to the element (C99 6.7.3p8), but the decayed
  // pointer does not carry the count. Put it back on the pointer.
  PVD->setType(Context.getCountAttributedType(
      PVD->getType(), CATy->getCountExpr(), CATy->isCountInBytes(),
      CATy->isOrNull(), CATy->getCoupledDecls()));
}

/// Whether the count \p E in a declaration of \p FD names a parameter of a
/// function, block or method whose body encloses that declaration.
static bool namesEnclosingParam(Expr *E, const FunctionDecl *FD) {
  bool IsDeref;
  const DeclRefExpr *DRE = Sema::getCountDeclRef(E, IsDeref);
  const auto *PVD = DRE ? dyn_cast<ParmVarDecl>(DRE->getDecl()) : nullptr;
  if (!PVD)
    return false;
  // The parameters of a prototype written in that body, such as a typedef's,
  // have the same context, so check that this is one of the body's own.
  const DeclContext *DC = PVD->getDeclContext();
  ArrayRef<ParmVarDecl *> Params;
  if (const auto *F = dyn_cast<FunctionDecl>(DC))
    Params = F->parameters();
  else if (const auto *B = dyn_cast<BlockDecl>(DC))
    Params = B->parameters();
  else if (const auto *M = dyn_cast<ObjCMethodDecl>(DC))
    Params = M->parameters();
  return llvm::is_contained(Params, PVD) &&
         DC->LexicallyEncloses(FD->getLexicalDeclContext());
}

/// Whether \p New and \p Old, counts in two declarations \p NewFD and \p OldFD
/// of a function, name the same thing. A parameter compares by its position
/// and type, so the counts of both declarations can name their own parameters,
/// or those of the typedef they were declared with, if those have the same
/// type, qualifiers included. A parameter of an enclosing body is not in
/// either function's parameter list, so it compares by identity.
static bool isSameCount(const ASTContext &Ctx, Expr *New, Expr *Old,
                        const FunctionDecl *NewFD, const FunctionDecl *OldFD) {
  if (!namesEnclosingParam(New, NewFD) && !namesEnclosingParam(Old, OldFD))
    return Ctx.hasSameExpr(New, Old);
  bool NewIsDeref, OldIsDeref;
  const DeclRefExpr *NewDRE = Sema::getCountDeclRef(New, NewIsDeref);
  const DeclRefExpr *OldDRE = Sema::getCountDeclRef(Old, OldIsDeref);
  return NewDRE && OldDRE && NewIsDeref == OldIsDeref &&
         NewDRE->getDecl() == OldDRE->getDecl();
}

/// Of \p New and \p Old, the types of one parameter in two declarations
/// \p NewFD and \p OldFD of a function, the count at the outermost pointer
/// level where they differ, or null if they agree. Only pointers are followed,
/// so counts below an _Atomic pointer or in a callback's parameters are not
/// compared.
static const CountAttributedType *
findConflictingCount(const ASTContext &Ctx, QualType New, QualType Old,
                     const FunctionDecl *NewFD, const FunctionDecl *OldFD) {
  for (; New->isPointerType() && Old->isPointerType();
       New = New->getPointeeType(), Old = Old->getPointeeType()) {
    const auto *NewCATy = New->getAs<CountAttributedType>();
    const auto *OldCATy = Old->getAs<CountAttributedType>();
    if (!NewCATy && !OldCATy)
      continue;
    if (!NewCATy || !OldCATy)
      return NewCATy ? NewCATy : OldCATy;

    if (NewCATy->isOrNull() != OldCATy->isOrNull())
      return NewCATy;
    // A count of one-byte elements is a size.
    if (NewCATy->isCountInBytes() != OldCATy->isCountInBytes()) {
      QualType Pointee = NewCATy->getPointeeType();
      if (!Pointee->isVoidType() && (Pointee->isIncompleteType() ||
                                     !Ctx.getTypeSizeInChars(Pointee).isOne()))
        return NewCATy;
    }
    assert(NewCATy->getCountExpr() && OldCATy->getCountExpr());
    if (!isSameCount(Ctx, NewCATy->getCountExpr(), OldCATy->getCountExpr(),
                     NewFD, OldFD))
      return NewCATy;
  }
  return nullptr;
}

bool Sema::CheckCountAttributedRedeclaration(const FunctionDecl *New,
                                             const FunctionDecl *Old) {
  for (unsigned I = 0, E = std::min(New->getNumParams(), Old->getNumParams());
       I != E; ++I) {
    const ParmVarDecl *NewParam = New->getParamDecl(I);
    if (const CountAttributedType *CATy =
            findConflictingCount(Context, NewParam->getType(),
                                 Old->getParamDecl(I)->getType(), New, Old)) {
      Diag(NewParam->getBeginLoc(),
           diag::err_count_attr_conflicting_redeclaration)
          << CATy->getKind();
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

  if (PointeeTy->isVoidType())
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
