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
#include "clang/Basic/SourceManager.h"
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
  if (PointeeTy->isIncompletableIncompleteType() && !CountInBytes) {
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
    // * When the pointee type is a "completable incomplete" type (i.e.
    // a type that is currently incomplete but might be completed later
    // on in the translation unit) the attribute is allowed by this method
    // but later uses of the FieldDecl are checked that the pointee type
    // is complete see `BoundsSafetyCheckAssignmentToCountAttrPtr`,
    // `BoundsSafetyCheckInitialization`, and
    // `BoundsSafetyCheckUseOfCountAttrPtr`
    //
    // * When the pointee type is a "incompletable incomplete" type (e.g.
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

SourceRange Sema::BoundsSafetySourceRangeFor(const CountAttributedType *CATy) {
  // Note: This implementation relies on `CountAttributedType` being unique.
  // E.g.:
  //
  // struct Foo {
  //   int count;
  //   char* __counted_by(count) buffer;
  //   char* __counted_by(count) buffer2;
  // };
  //
  // The types of `buffer` and `buffer2` are unique. The types being
  // unique means the SourceLocation of the `counted_by` expression can be used
  // to find where the attribute was written.

  auto Fallback = CATy->getCountExpr()->getSourceRange();
  auto CountExprBegin = CATy->getCountExpr()->getBeginLoc();

  // FIXME: We currently don't support the count expression being a macro
  // itself. E.g.:
  //
  // #define ZERO 0
  // int* __counted_by(ZERO) x;
  //
  if (SourceMgr.isMacroBodyExpansion(CountExprBegin))
    return Fallback;

  auto FetchIdentifierTokenFromOffset =
      [&](ssize_t Offset) -> std::optional<Token> {
    SourceLocation OffsetLoc = CountExprBegin.getLocWithOffset(Offset);
    Token Result;
    if (Lexer::getRawToken(OffsetLoc, Result, SourceMgr, getLangOpts()))
      return std::optional<Token>(); // Failed

    if (!Result.isAnyIdentifier())
      return std::optional<Token>(); // Failed

    return Result; // Success
  };

  auto CountExprEnd = CATy->getCountExpr()->getEndLoc();
  auto FindRParenTokenAfter = [&]() -> std::optional<Token> {
    auto CountExprEndSpelling = SourceMgr.getSpellingLoc(CountExprEnd);
    auto MaybeRParenTok = Lexer::findNextToken(
        CountExprEndSpelling, getSourceManager(), getLangOpts());

    if (!MaybeRParenTok.has_value())
      return std::nullopt;

    if (!MaybeRParenTok->is(tok::r_paren))
      return std::nullopt;

    return *MaybeRParenTok;
  };

  // Step back two characters to point at the last character of the attribute
  // text.
  //
  // __counted_by(count)
  //            ^ ^
  //            | |
  //            ---
  auto MaybeLastAttrCharToken = FetchIdentifierTokenFromOffset(-2);
  if (!MaybeLastAttrCharToken)
    return Fallback;

  auto LastAttrCharToken = MaybeLastAttrCharToken.value();

  if (LastAttrCharToken.getLength() > 1) {
    // Special case: When the character is part of a macro the Token we get
    // is the whole macro name (e.g. `__counted_by`).
    if (LastAttrCharToken.getRawIdentifier() !=
        CATy->GetAttributeName(/*WithMacroPrefix=*/true))
      return Fallback;

    // Found the beginning of the `__counted_by` macro
    SourceLocation Begin = LastAttrCharToken.getLocation();
    // Now try to find the closing `)` of the macro.
    auto MaybeRParenTok = FindRParenTokenAfter();
    if (!MaybeRParenTok.has_value())
      return Fallback;

    return SourceRange(Begin, MaybeRParenTok->getLocation());
  }

  assert(LastAttrCharToken.getLength() == 1);
  // The Token we got back is just the last character of the identifier.
  // This means a macro is not being used and instead the attribute is being
  // used directly. We need to find the beginning of the identifier. We support
  // two cases:
  //
  // * Non-affixed version. E.g: `counted_by`
  // * Affixed version. E.g.: `__counted_by__`

  // Try non-affixed version. E.g.:
  //
  // __attribute__((counted_by(count)))
  //                ^          ^
  //                |          |
  //                ------------

  // +1 is for `(`
  const ssize_t NonAffixedSkipCount =
      CATy->GetAttributeName(/*WithMacroPrefix=*/false).size() + 1;
  auto MaybeNonAffixedBeginToken =
      FetchIdentifierTokenFromOffset(-NonAffixedSkipCount);
  if (!MaybeNonAffixedBeginToken)
    return Fallback;

  auto NonAffixedBeginToken = MaybeNonAffixedBeginToken.value();
  if (NonAffixedBeginToken.getRawIdentifier() ==
      CATy->GetAttributeName(/*WithMacroPrefix=*/false)) {
    // Found the beginning of the `counted_by`-like attribute
    auto SL = NonAffixedBeginToken.getLocation();

    // Now try to find the closing `)` of the attribute
    auto MaybeRParenTok = FindRParenTokenAfter();
    if (!MaybeRParenTok.has_value())
      return Fallback;

    return SourceRange(SL, MaybeRParenTok->getLocation());
  }

  // Try affixed version. E.g.:
  //
  // __attribute__((__counted_by__(count)))
  //                ^              ^
  //                |              |
  //                ----------------
  std::string AffixedTokenStr =
      (llvm::Twine("__") + CATy->GetAttributeName(/*WithMacroPrefix=*/false) +
       llvm::Twine("__"))
          .str();
  // +1 is for `(`
  // +4 is for the 4 `_` characters
  const ssize_t AffixedSkipCount =
      CATy->GetAttributeName(/*WithMacroPrefix=*/false).size() + 1 + 4;
  auto MaybeAffixedBeginToken =
      FetchIdentifierTokenFromOffset(-AffixedSkipCount);
  if (!MaybeAffixedBeginToken)
    return Fallback;

  auto AffixedBeginToken = MaybeAffixedBeginToken.value();
  if (AffixedBeginToken.getRawIdentifier() != AffixedTokenStr)
    return Fallback;

  // Found the beginning of the `__counted_by__`-like like attribute.
  auto SL = AffixedBeginToken.getLocation();
  // Now try to find the closing `)` of the attribute
  auto MaybeRParenTok = FindRParenTokenAfter();
  if (!MaybeRParenTok.has_value())
    return Fallback;

  return SourceRange(SL, MaybeRParenTok->getLocation());
}

static void EmitIncompleteCountedByPointeeNotes(Sema &S,
                                                const CountAttributedType *CATy,
                                                NamedDecl *IncompleteTyDecl,
                                                bool NoteAttrLocation = true) {
  assert(IncompleteTyDecl == nullptr || isa<TypeDecl>(IncompleteTyDecl));

  if (NoteAttrLocation) {
    // Note where the attribute is declared
    auto AttrSrcRange = S.BoundsSafetySourceRangeFor(CATy);
    S.Diag(AttrSrcRange.getBegin(), diag::note_named_attribute)
        << CATy->GetAttributeName(/*WithMacroPrefix=*/true) << AttrSrcRange;
  }

  if (!IncompleteTyDecl)
    return;

  // If there's an associated forward declaration display it to emphasize
  // why the type is incomplete (all we have is a forward declaration).

  // Note the `IncompleteTyDecl` type is the underlying type which might not
  // be the same as `CATy->getPointeeType()` which could be a typedef.
  //
  // The diagnostic printed will be at the location of the underlying type but
  // the diagnostic text will print the type of `CATy->getPointeeType()` which
  // could be a typedef name rather than the underlying type. This is ok
  // though because the diagnostic will print the underlying type name too.
  // E.g:
  //
  // `forward declaration of 'Incomplete_Struct_t'
  //  (aka 'struct IncompleteStructTy')`
  //
  // If this ends up being confusing we could emit a second diagnostic (one
  // explaining where the typedef is) but that seems overly verbose.

  S.Diag(IncompleteTyDecl->getBeginLoc(), diag::note_forward_declaration)
      << CATy->getPointeeType();
}

static bool
HasCountedByAttrOnIncompletePointee(QualType Ty, NamedDecl **ND,
                                    const CountAttributedType **CATyOut,
                                    QualType *PointeeTyOut) {
  auto *CATy = Ty->getAs<CountAttributedType>();
  if (!CATy)
    return false;

  // Incomplete pointee type is only a problem for
  // counted_by/counted_by_or_null
  if (CATy->isCountInBytes())
    return false;

  auto PointeeTy = CATy->getPointeeType();
  if (PointeeTy.isNull())
    return false; // Reachable?

  if (!PointeeTy->isIncompleteType(ND))
    return false;

  if (CATyOut)
    *CATyOut = CATy;
  if (PointeeTyOut)
    *PointeeTyOut = PointeeTy;
  return true;
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
/// \param ComputeAssignee If provided this function will be called before
///        emitting a diagnostic. The function should return the name of
///        entity being assigned to or an empty string if this cannot be
///        determined.
///
/// \returns True iff no diagnostic where emitted, false otherwise.
static bool BoundsSafetyCheckAssignmentToCountAttrPtrWithIncompletePointeeTy(
    Sema &S, QualType LHSTy, Expr *RHSExpr, Sema::AssignmentAction Action,
    SourceLocation Loc, std::function<std::string()> ComputeAssignee) {
  NamedDecl *IncompleteTyDecl = nullptr;
  const CountAttributedType *CATy = nullptr;
  QualType PointeeTy;
  if (!HasCountedByAttrOnIncompletePointee(LHSTy, &IncompleteTyDecl, &CATy,
                                           &PointeeTy))
    return true;
  assert(CATy && !CATy->isCountInBytes() && !PointeeTy.isNull());

  // It's not expected that the diagnostic be emitted in these cases.
  // It's not necessarily a problem but we should catch when this starts
  // to happen.
  assert(Action != Sema::AA_Converting && Action != Sema::AA_Sending &&
         Action != Sema::AA_Casting && Action != Sema::AA_Passing_CFAudited);

  // By having users provide a function we only pay the cost of allocation the
  // string and computing when a diagnostic is emitted.
  std::string Assignee = ComputeAssignee ? ComputeAssignee() : "";
  {
    auto D = S.Diag(Loc, diag::err_counted_by_on_incomplete_type_on_assign)
             << /*0*/ (int)Action << /*1*/ Assignee
             << /*2*/ (Assignee.size() > 0)
             << /*3*/ isa<ImplicitValueInitExpr>(RHSExpr) << /*4*/ LHSTy
             << /*5*/ CATy->GetAttributeName(/*WithMacroPrefix=*/true)
             << /*6*/ PointeeTy << /*7*/ CATy->isOrNull();

    if (RHSExpr->getSourceRange().isValid())
      D << RHSExpr->getSourceRange();
  }

  EmitIncompleteCountedByPointeeNotes(S, CATy, IncompleteTyDecl);
  return false; // check failed
}

bool Sema::BoundsSafetyCheckAssignmentToCountAttrPtr(
    QualType LHSTy, Expr *RHSExpr, Sema::AssignmentAction Action,
    SourceLocation Loc, std::function<std::string()> ComputeAssignee) {
  return BoundsSafetyCheckAssignmentToCountAttrPtrWithIncompletePointeeTy(
      *this, LHSTy, RHSExpr, Action, Loc, ComputeAssignee);
}

bool Sema::BoundsSafetyCheckInitialization(const InitializedEntity &Entity,
                                           const InitializationKind &Kind,
                                           Sema::AssignmentAction Action,
                                           QualType LHSType, Expr *RHSExpr) {
  bool ChecksPassed = true;
  auto SL = Kind.getLocation();

  // Note: We don't call `BoundsSafetyCheckAssignmentToCountAttrPtr` here
  // because we need conditionalize what is checked. In downstream
  // Clang `counted_by` is supported on variable definitions and in that
  // implementation an error diagnostic will be emitted on the variable
  // definition if the pointee is an incomplete type. To avoid warning about the
  // same problem twice (once when the variable is defined, once when Sema
  // checks the initializer) we skip checking the initializer if it's a
  // variable.
  if (Action == Sema::AA_Initializing &&
      Entity.getKind() != InitializedEntity::EK_Variable) {

    if (!BoundsSafetyCheckAssignmentToCountAttrPtrWithIncompletePointeeTy(
            *this, LHSType, RHSExpr, Action, SL, [&Entity]() -> std::string {
              if (const auto *VD =
                      dyn_cast_or_null<ValueDecl>(Entity.getDecl())) {
                return VD->getQualifiedNameAsString();
              }
              return "";
            })) {
      ChecksPassed = false;

      // It's not necessarily bad if this assert fails but we should catch
      // if this happens.
      assert(Entity.getKind() == InitializedEntity::EK_Member);
    }
  }

  return ChecksPassed;
}

static bool BoundsSafetyCheckUseOfCountAttrPtrWithIncompletePointeeTy(Sema &S,
                                                                      Expr *E) {
  QualType T = E->getType();
  assert(T->isPointerType());

  const CountAttributedType *CATy = nullptr;
  QualType PointeeTy;
  NamedDecl *IncompleteTyDecl = nullptr;
  if (!HasCountedByAttrOnIncompletePointee(T, &IncompleteTyDecl, &CATy,
                                           &PointeeTy))
    return true;
  assert(CATy && !CATy->isCountInBytes() && !PointeeTy.isNull());

  // Generate a string for the diagnostic that describes the "use".
  // The string is specialized for direct calls to produce a better
  // diagnostic.
  StringRef DirectCallFn;
  std::string UseStr;
  if (const auto *CE = dyn_cast<CallExpr>(E->IgnoreParens())) {
    if (const auto *FD = CE->getDirectCallee()) {
      DirectCallFn = FD->getName();
    }
  }
  int SelectExprKind = DirectCallFn.size() > 0 ? 1 : 0;
  if (SelectExprKind) {
    UseStr = DirectCallFn;
  } else {
    llvm::raw_string_ostream SS(UseStr);
    E->printPretty(SS, nullptr, PrintingPolicy(S.getPrintingPolicy()));
  }
  assert(UseStr.size() > 0);

  S.Diag(E->getBeginLoc(), diag::err_counted_by_on_incomplete_type_on_use)
      << /*0*/ SelectExprKind << /*1*/ UseStr << /*2*/ T << /*3*/ PointeeTy
      << /*4*/ CATy->GetAttributeName(/*WithMacroPrefix=*/true)
      << /*5*/ CATy->isOrNull() << E->getSourceRange();

  EmitIncompleteCountedByPointeeNotes(S, CATy, IncompleteTyDecl);
  return false;
}

bool Sema::BoundsSafetyCheckUseOfCountAttrPtr(Expr *E) {

  QualType T = E->getType();
  if (!T->isPointerType())
    return true;

  return BoundsSafetyCheckUseOfCountAttrPtrWithIncompletePointeeTy(*this, E);
}

} // namespace clang
