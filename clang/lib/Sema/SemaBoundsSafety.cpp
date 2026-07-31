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
#include "TreeTransform.h"
#include "clang/Lex/Lexer.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/SaveAndRestore.h"
#include <optional>

namespace clang {

static CountAttributedType::DynamicCountPointerKind
getCountAttrKind(bool CountInBytes, bool OrNull) {
  if (CountInBytes)
    return OrNull ? CountAttributedType::SizedByOrNull
                  : CountAttributedType::SizedBy;
  return OrNull ? CountAttributedType::CountedByOrNull
                : CountAttributedType::CountedBy;
}

namespace {
struct BoundsAttrFlags {
  bool CountInBytes = false;
  bool OrNull = false;
};
} // namespace

static std::optional<BoundsAttrFlags> getBoundsAttrFlags(ParsedAttr::Kind K) {
  BoundsAttrFlags Flags;
  switch (K) {
  case ParsedAttr::AT_CountedBy:
    break;
  case ParsedAttr::AT_CountedByOrNull:
    Flags.OrNull = true;
    break;
  case ParsedAttr::AT_SizedBy:
    Flags.CountInBytes = true;
    break;
  case ParsedAttr::AT_SizedByOrNull:
    Flags.CountInBytes = true;
    Flags.OrNull = true;
    break;
  default:
    return std::nullopt;
  }
  return Flags;
}

bool Sema::ActOnLateParsedTypeAttr(ParsedAttr::Kind AttrKind,
                                   SourceLocation AttrNameLoc, QualType &type,
                                   LateParsedTypeAttribute *LTA) {
  // Not an attribute this mechanism handles; leave it alone.
  std::optional<BoundsAttrFlags> Flags = getBoundsAttrFlags(AttrKind);
  if (!Flags)
    return false;

  // The argument-independent validation, run before the attribute's argument
  // has been parsed. An array parameter has already decayed to a pointer, so
  // unlike a flexible array member there is no array spelling to accept here.
  if (!type->isPointerType()) {
    unsigned Kind = getCountAttrKind(Flags->CountInBytes, Flags->OrNull);
    Diag(AttrNameLoc, diag::err_count_attr_not_on_ptr_or_flexible_array_member)
        << Kind << AttrNameLoc << /*do not suggest counted_by*/ 0;
    return false;
  }

  type = Context.getLateParsedAttrType(type, LTA);
  return true;
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

/// Diagnose a counted_by-family attribute whose pointee (or array element)
/// type \p PointeeTy cannot support bounds computation. Shared between the
/// struct-field and function-parameter paths so both accept and reject the
/// same shapes. \p DowngradeFAMPointeeErrToWarn is the Linux-kernel
/// workaround; see the caller in CheckCountedByAttrOnField. Returns true if
/// the attribute cannot be applied.
static bool CheckCountAttrPointeeType(Sema &S, QualType PointeeTy,
                                      bool CountInBytes, unsigned Kind,
                                      int SelectPtrOrArr,
                                      bool DowngradeFAMPointeeErrToWarn,
                                      SourceLocation Loc, SourceRange Range) {
  CountedByInvalidPointeeTypeKind InvalidTypeKind =
      CountedByInvalidPointeeTypeKind::VALID;
  bool ShouldWarn = false;
  if (!CountInBytes && PointeeTy->isAlwaysIncompleteType()) {
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
    //   size_t count;
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
    // but later uses of the annotated declaration are checked that the pointee
    // type is complete see `BoundsSafetyCheckAssignmentToCountAttrPtr`,
    // `BoundsSafetyCheckInitialization`, and
    // `BoundsSafetyCheckUseOfCountAttrPtr`
    //
    // * When the pointee type is always an incomplete type (e.g.
    // `void` in strict C mode) the attribute is disallowed by this method
    // because we know the type can never be completed so there's no reason
    // to allow it.
    //
    // Exception: void has an implicit size of 1 byte for pointer arithmetic
    // (following GNU convention). Therefore, counted_by on void* is allowed
    // and behaves equivalently to sized_by (treating the count as bytes).
    bool IsVoidPtr = PointeeTy->isVoidType();
    if (IsVoidPtr) {
      // Emit a warning that this is a GNU extension.
      S.Diag(Loc, diag::ext_gnu_counted_by_void_ptr) << Kind;
      S.Diag(Loc, diag::note_gnu_counted_by_void_ptr_use_sized_by) << Kind;
      assert(InvalidTypeKind == CountedByInvalidPointeeTypeKind::VALID);
    } else {
      InvalidTypeKind = CountedByInvalidPointeeTypeKind::INCOMPLETE;
    }
  } else if (PointeeTy->isSizelessType()) {
    InvalidTypeKind = CountedByInvalidPointeeTypeKind::SIZELESS;
  } else if (PointeeTy->isFunctionType()) {
    InvalidTypeKind = CountedByInvalidPointeeTypeKind::FUNCTION;
  } else if (!CountInBytes &&
             PointeeTy->isStructureTypeWithFlexibleArrayMember()) {
    if (DowngradeFAMPointeeErrToWarn)
      ShouldWarn = true;
    InvalidTypeKind = CountedByInvalidPointeeTypeKind::FLEXIBLE_ARRAY_MEMBER;
  }

  if (InvalidTypeKind != CountedByInvalidPointeeTypeKind::VALID) {
    unsigned DiagID = ShouldWarn
                          ? diag::warn_counted_by_attr_elt_type_unknown_size
                          : diag::err_counted_by_attr_pointee_unknown_size;
    S.Diag(Loc, DiagID) << SelectPtrOrArr << PointeeTy << (int)InvalidTypeKind
                        << (ShouldWarn ? 1 : 0) << Kind << Range;
    return true;
  }
  return false;
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
  //
  // Downgrading the FAM-pointee error to a warning on such arrays is a
  // workaround for the Linux kernel that has already adopted `counted_by` on
  // a FAM where the pointee is a struct with a FAM. This should be an error
  // because computing the bounds of the array cannot be done correctly
  // without manually traversing every struct object in the array at runtime.
  // To allow the code to be built the error is downgraded to a warning.
  bool DowngradeFAMPointeeErrToWarn =
      FieldTy->isArrayType() && !getLangOpts().BoundsSafety;
  if (CheckCountAttrPointeeType(*this, PointeeTy, CountInBytes, Kind,
                                SelectPtrOrArr, DowngradeFAMPointeeErrToWarn,
                                FD->getBeginLoc(), FD->getSourceRange()))
    return true;

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

/// Build the CountAttributedType for a counted_by-family attribute on a
/// parameter's type, whose argument has just been parsed. \p Params is the
/// prototype's parameter list; the count must name one of them. Null on error.
static QualType buildCountAttributedTypeForParam(Sema &S, QualType InnerTy,
                                                 ArrayRef<Decl *> Params,
                                                 ParsedAttr &AL) {
  std::optional<BoundsAttrFlags> Flags = getBoundsAttrFlags(AL.getKind());
  assert(Flags && "placeholder for a non-counted_by-family attribute");
  auto [CountInBytes, OrNull] = *Flags;
  unsigned Kind = getCountAttrKind(CountInBytes, OrNull);

  // The same pointee rules as the struct-field path, checked in the same
  // order: the pointee's size before the count expression. A placeholder is
  // only created for a pointer, so the pointee is always there to take.
  if (CheckCountAttrPointeeType(S, InnerTy->getPointeeType(), CountInBytes,
                                Kind, /*SelectPtrOrArr=*/0,
                                /*DowngradeFAMPointeeErrToWarn=*/false,
                                AL.getLoc(), AL.getRange()))
    return QualType();

  Expr *CountExpr = AL.getArgAsExpr(0);
  if (!CountExpr)
    return QualType();

  // The argument must name a single declaration, so that assignments to the
  // count can be related back to the pointer. No paren-stripping: the
  // expression handed to BuildCountAttributedArrayOrPointerType must itself be
  // the DeclRefExpr (BuildTypeCoupledDecls casts it), and the field path in
  // CheckCountedByAttrOnField rejects parens the same way.
  auto *DRE = dyn_cast<DeclRefExpr>(CountExpr);
  if (!DRE) {
    S.Diag(CountExpr->getBeginLoc(),
           diag::err_count_attr_only_support_simple_decl_reference)
        << Kind << CountExpr->getSourceRange();
    return QualType();
  }

  // The count must name a parameter of *this* prototype. Check list membership
  // rather than DeclContext: mid-parse every ParmVarDecl shares the enclosing
  // context, so a DeclContext check would wrongly accept an enclosing
  // prototype's parameter, as in
  //   void f(int n, void (*cb)(int *__counted_by(n) p));
  auto *CountDecl = dyn_cast<ParmVarDecl>(DRE->getDecl());
  if (!CountDecl || !llvm::is_contained(Params, CountDecl)) {
    S.Diag(DRE->getBeginLoc(), diag::err_count_attr_not_param_of_same_function)
        << DRE->getDecl() << Kind << DRE->getSourceRange();
    return QualType();
  }

  if (!CountDecl->getType()->isIntegerType() ||
      CountDecl->getType()->isBooleanType()) {
    S.Diag(DRE->getBeginLoc(), diag::err_count_attr_argument_not_integer)
        << Kind << DRE->getSourceRange();
    return QualType();
  }

  return S.BuildCountAttributedArrayOrPointerType(InnerTy, CountExpr,
                                                  CountInBytes, OrNull);
}

namespace {

/// Where a LateParsedAttrType placeholder sits inside a parameter's type.
enum class LateAttrPosition {
  Parameter,
  NestedPointer,
  ArrayElement,
};

using LateAttrTypeAndPosition =
    std::pair<const LateParsedAttrType *, LateAttrPosition>;

} // namespace

/// Collect every LateParsedAttrType written in \p T, together with where it
/// sits. A placeholder must never survive into the finalized AST, so this has
/// to reach all of them, not just the ones in a position we can resolve.
static void
findLateParsedAttrTypes(ASTContext &Ctx, QualType T, LateAttrPosition Pos,
                        SmallVectorImpl<LateAttrTypeAndPosition> &Out) {
  // A position is only ever demoted: once a placeholder is out of the running
  // for describing the parameter it cannot come back into it.
  auto Demote = [](LateAttrPosition Pos, LateAttrPosition To) {
    return Pos == LateAttrPosition::Parameter ? To : Pos;
  };

  // Each step below matches the node itself with isa/dyn_cast and desugars by
  // a single step otherwise. Desugaring helpers that use getAs would step
  // straight over a placeholder, which is the one thing this must not do; the
  // getPointeeType() call is safe only because the isa<> in front of it
  // guarantees the match is the node itself, not something behind sugar.
  while (!T.isNull()) {
    const Type *Ty = T.getTypePtr();

    if (const auto *LPT = dyn_cast<LateParsedAttrType>(Ty)) {
      Out.emplace_back(LPT, Pos);
      T = LPT->desugar();
      continue;
    }

    // The two pointer kinds a declarator chunk can produce in C. Late parsing
    // is not enabled for C++, so references and member pointers cannot occur.
    if (isa<PointerType, BlockPointerType>(Ty)) {
      T = Ty->getPointeeType();
      Pos = Demote(Pos, LateAttrPosition::NestedPointer);
      continue;
    }
    if (const auto *AT = dyn_cast<AtomicType>(Ty)) {
      // Not sugar, so the desugar fallback would stop here. No demotion: a
      // placeholder cannot sit directly below _Atomic, since an atomic pointer
      // is rejected when the placeholder is created.
      T = AT->getValueType();
      continue;
    }
    if (const auto *AT = dyn_cast<ArrayType>(Ty)) {
      T = AT->getElementType();
      Pos = Demote(Pos, LateAttrPosition::ArrayElement);
      continue;
    }
    if (const auto *FT = dyn_cast<FunctionType>(Ty)) {
      // A count written inside a function prototype names that prototype's own
      // parameters, so the position resets: the return type and each parameter
      // type are "own types" again, just of a different declaration.
      // RebuildTypeWithLateParsedAttr::TransformFunctionProtoType switches the
      // parameter list to match.
      //
      // Only the return type reaches this in practice: the parser resolves at
      // every prototype's closing paren, so a nested prototype's parameters
      // were handled by their own trigger. The loop below keeps the walk
      // exhaustive, since a placeholder it misses escapes into the AST.
      findLateParsedAttrTypes(Ctx, FT->getReturnType(),
                              LateAttrPosition::Parameter, Out);
      if (const auto *FPT = dyn_cast<FunctionProtoType>(FT))
        for (QualType ParamTy : FPT->getParamTypes())
          findLateParsedAttrTypes(Ctx, ParamTy, LateAttrPosition::Parameter,
                                  Out);
      return;
    }

    // Anything else can only hide a placeholder behind sugar.
    QualType Desugared = T.getSingleStepDesugaredType(Ctx);
    if (Desugared == T)
      return;
    T = Desugared;
  }
}

namespace {

/// Rebuilds a type, replacing each LateParsedAttrType placeholder with the
/// concrete type its attribute denotes, parsing the cached tokens on the way.
///
/// Every placeholder is replaced, including the ones whose attribute turns out
/// to be unusable: a placeholder may not reach the finalized AST, and once its
/// tokens are parsed it no longer refers to a live attribute either.
struct RebuildTypeWithLateParsedAttr
    : TreeTransform<RebuildTypeWithLateParsedAttr> {
  ParmVarDecl *PVD;
  ArrayRef<Decl *> Params;
  Sema::ParseLateParsedTypeAttributeCB *ParseCallback;
  ArrayRef<LateAttrTypeAndPosition> Positions;

  RebuildTypeWithLateParsedAttr(Sema &SemaRef, ParmVarDecl *PVD,
                                ArrayRef<Decl *> Params,
                                Sema::ParseLateParsedTypeAttributeCB *ParseCB,
                                ArrayRef<LateAttrTypeAndPosition> Positions)
      : TreeTransform(SemaRef), PVD(PVD), Params(Params),
        ParseCallback(ParseCB), Positions(Positions) {}

  /// findLateParsedAttrTypes walked the same type this transform is walking, so
  /// every placeholder reached here was classified by it.
  LateAttrPosition getPosition(const LateParsedAttrType *LPT) const {
    for (LateAttrTypeAndPosition Entry : Positions)
      if (Entry.first == LPT)
        return Entry.second;
    llvm_unreachable("placeholder reached by the transform was not classified");
  }

  // TransformFunctionProtoType is overloaded; overriding one would hide the
  // rest. The base two-argument version dispatches to the five-argument one
  // through getDerived().
  using TreeTransform::TransformFunctionProtoType;

  /// A count inside a nested prototype names *that* prototype's parameters, as
  /// in `void f(int *__counted_by(len) (*cb)(int len), int len2);`. Two things
  /// are needed and they are not alternatives: the re-entered scope makes `len`
  /// findable at all (the inner prototype's scope was popped when its
  /// declarator finished), and the parameter list decides whether what lookup
  /// found is allowed (lookup falls through to the enclosing scopes).
  QualType TransformFunctionProtoType(TypeLocBuilder &TLB,
                                      FunctionProtoTypeLoc TL) {
    SmallVector<Decl *, 4> InnerParams;
    for (unsigned I = 0, E = TL.getNumParams(); I != E; ++I)
      if (ParmVarDecl *PD = TL.getParam(I))
        InnerParams.push_back(PD);

    Sema::FunctionPrototypeScopeRAII ProtoScope(SemaRef, InnerParams);
    SaveAndRestore<ArrayRef<Decl *>> SavedParams(Params, InnerParams);
    return TreeTransform::TransformFunctionProtoType(TLB, TL);
  }

  /// A no-prototype function declares no parameters, so nothing in its return
  /// type can name one: `void f(int *__counted_by(len) (*cb)(), int len);`.
  /// Clearing the list is what rejects that; otherwise the list is still the
  /// enclosing prototype's and `len` is accepted. There is no scope worth
  /// re-entering: an empty one would not change what lookup finds.
  QualType TransformFunctionNoProtoType(TypeLocBuilder &TLB,
                                        FunctionNoProtoTypeLoc TL) {
    SaveAndRestore<ArrayRef<Decl *>> SavedParams(Params, ArrayRef<Decl *>());
    return TreeTransform::TransformFunctionNoProtoType(TLB, TL);
  }

  QualType TransformLateParsedAttrType(TypeLocBuilder &TLB,
                                       LateParsedAttrTypeLoc TL) {
    const LateParsedAttrType *LPT = TL.getTypePtr();
    LateParsedTypeAttribute *LTA = LPT->getLateParsedAttribute();
    assert(LTA && "LateParsedAttrType without a LateParsedTypeAttribute");

    AttributeFactory AF;
    ParsedAttributes Attrs(AF);

    // Parse the cached tokens. The callback also destroys LTA, so from here on
    // the placeholder refers to an attribute that no longer exists.
    assert(ParseCallback);
    ParseCallback(LTA, &Attrs);

    QualType InnerTy = TransformType(TLB, TL.getInnerLoc());
    if (InnerTy.isNull()) {
      PVD->setInvalidDecl();
      return QualType();
    }

    // An empty list means the argument failed to parse, which is already
    // diagnosed.
    QualType T;
    if (!Attrs.empty()) {
      assert(Attrs.size() == 1);
      LateAttrPosition Pos = getPosition(LPT);
      if (Pos == LateAttrPosition::Parameter) {
        T = buildCountAttributedTypeForParam(SemaRef, InnerTy, Params,
                                             Attrs[0]);
      } else {
        std::optional<BoundsAttrFlags> Flags =
            getBoundsAttrFlags(Attrs[0].getKind());
        assert(Flags && "placeholder for a non-counted_by-family attribute");
        SemaRef.Diag(TL.getAttrNameLoc(), diag::err_count_attr_on_nested_type)
            << getCountAttrKind(Flags->CountInBytes, Flags->OrNull)
            << (Pos == LateAttrPosition::NestedPointer ? /*nested pointer*/ 0
                                                       : /*array element*/ 1);
      }
    }

    if (T.isNull()) {
      // Drop the attribute and keep the type it wrapped. Returning nothing
      // would leave the caller holding the original type, placeholder and all.
      PVD->setInvalidDecl();
      return InnerTy;
    }

    TLB.push<CountAttributedTypeLoc>(T);
    return T;
  }
};

} // namespace

Sema::FunctionPrototypeScopeRAII::FunctionPrototypeScopeRAII(
    Sema &S, ArrayRef<Decl *> Params)
    : S(S), ProtoScope(S.getCurScope(),
                       Scope::FunctionPrototypeScope | Scope::DeclScope,
                       S.getDiagnostics()) {
  S.CurScope = &ProtoScope;
  for (Decl *D : Params)
    S.ActOnReenterCXXMethodParameter(&ProtoScope,
                                     dyn_cast_if_present<ParmVarDecl>(D));
}

Sema::FunctionPrototypeScopeRAII::~FunctionPrototypeScopeRAII() {
  // ActOnPopScope is what takes the parameters back out of the IdResolver. It
  // is a no-op otherwise here: a scope holding only ParmVarDecls produces no
  // end-of-scope diagnostics.
  S.ActOnPopScope(SourceLocation(), &ProtoScope);
  S.CurScope = ProtoScope.getParent();
}

void Sema::ProcessLateParsedTypeAttributesForParams(
    ArrayRef<Decl *> Params, ParseLateParsedTypeAttributeCB *ParseCB) {
  // The parameters of a nested prototype are put back in scope by
  // RebuildTypeWithLateParsedAttr::TransformFunctionProtoType. Do the same for
  // the outermost prototype's own parameters rather than relying on the
  // caller's scope, so that resolution works wherever this is called from.
  //
  // Created on the first parameter that needs it: the parser calls this for
  // every prototype it parses under -fexperimental-late-parse-attributes, and
  // almost none of them carry a late-parsed attribute.
  std::optional<FunctionPrototypeScopeRAII> ProtoScope;

  for (Decl *D : Params) {
    auto *PVD = dyn_cast_if_present<ParmVarDecl>(D);
    if (!PVD || !PVD->getTypeSourceInfo())
      continue;

    TypeSourceInfo *OldTSI = PVD->getTypeSourceInfo();
    SmallVector<LateAttrTypeAndPosition, 2> Found;
    findLateParsedAttrTypes(Context, OldTSI->getType(),
                            LateAttrPosition::Parameter, Found);
    if (Found.empty())
      continue;

    if (!ProtoScope)
      ProtoScope.emplace(*this, Params);

    RebuildTypeWithLateParsedAttr Rebuild(*this, PVD, Params, ParseCB, Found);
    TypeSourceInfo *TSI = Rebuild.TransformType(OldTSI);
    if (!TSI) {
      PVD->setInvalidDecl();
      continue;
    }
    PVD->setTypeSourceInfo(TSI);
    // A parameter's declared type is the adjusted form of its written type.
    PVD->setType(Context.getAdjustedParameterType(TSI->getType()));
  }
}

} // namespace clang
