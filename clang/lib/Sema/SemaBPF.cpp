//===------ SemaBPF.cpp ---------- BPF target-specific routines -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis functions specific to BPF.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaBPF.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/APSInt.h"
#include <optional>

namespace clang {

SemaBPF::SemaBPF(Sema &S) : SemaBase(S) {}

static bool isValidPreserveFieldInfoArg(Expr *Arg) {
  if (Arg->getType()->getAsPlaceholderType())
    return false;

  // The first argument needs to be a record field access.
  // If it is an array element access, we delay decision
  // to BPF backend to check whether the access is a
  // field access or not.
  return (Arg->IgnoreParens()->getObjectKind() == OK_BitField ||
          isa<MemberExpr>(Arg->IgnoreParens()) ||
          isa<ArraySubscriptExpr>(Arg->IgnoreParens()));
}

static bool isValidPreserveTypeInfoArg(Expr *Arg) {
  QualType ArgType = Arg->getType();
  if (ArgType->getAsPlaceholderType())
    return false;

  // for TYPE_EXISTENCE/TYPE_MATCH/TYPE_SIZEOF reloc type
  // format:
  //   1. __builtin_preserve_type_info(*(<type> *)0, flag);
  //   2. <type> var;
  //      __builtin_preserve_type_info(var, flag);
  if (!isa<DeclRefExpr>(Arg->IgnoreParens()) &&
      !isa<UnaryOperator>(Arg->IgnoreParens()))
    return false;

  // Typedef type.
  if (ArgType->getAs<TypedefType>())
    return true;

  // Record type or Enum type.
  if (const auto *RT = ArgType->getAsCanonical<TagType>())
    if (!RT->getDecl()->getDeclName().isEmpty())
      return true;

  return false;
}

static bool isValidPreserveEnumValueArg(Expr *Arg) {
  QualType ArgType = Arg->getType();
  if (ArgType->getAsPlaceholderType())
    return false;

  // for ENUM_VALUE_EXISTENCE/ENUM_VALUE reloc type
  // format:
  //   __builtin_preserve_enum_value(*(<enum_type> *)<enum_value>,
  //                                 flag);
  const auto *UO = dyn_cast<UnaryOperator>(Arg->IgnoreParens());
  if (!UO)
    return false;

  const auto *CE = dyn_cast<CStyleCastExpr>(UO->getSubExpr());
  if (!CE)
    return false;
  if (CE->getCastKind() != CK_IntegralToPointer &&
      CE->getCastKind() != CK_NullToPointer)
    return false;

  // The integer must be from an EnumConstantDecl.
  const auto *DR = dyn_cast<DeclRefExpr>(CE->getSubExpr());
  if (!DR)
    return false;

  const EnumConstantDecl *Enumerator =
      dyn_cast<EnumConstantDecl>(DR->getDecl());
  if (!Enumerator)
    return false;

  // The type must be EnumType.
  const auto *ED = ArgType->getAsEnumDecl();
  if (!ED)
    return false;

  // The enum value must be supported.
  return llvm::is_contained(ED->enumerators(), Enumerator);
}

// The kind of a C-style cast of the enumerator E to a pointer, as computed by
// Sema::PrepareScalarCast().
static CastKind getEnumeratorToPointerCastKind(ASTContext &Context,
                                               const Expr *E) {
  return E->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull)
             ? CK_NullToPointer
             : CK_IntegralToPointer;
}

// In __builtin_preserve_enum_value(*(<enum_type> *)<enum_value>, flag), the
// enumerator refers to the enumerator itself rather than its value. If it is
// wrapped by BuildPreserveAIEnumeratorRef(), use the original DeclRefExpr, so
// that the argument is the same as without the preserve_access_index attribute.
static void unwrapPreserveEnumValueArg(ASTContext &Context, Expr *Arg) {
  auto *UO = dyn_cast<UnaryOperator>(Arg->IgnoreParens());
  if (!UO || UO->getOpcode() != UO_Deref)
    return;
  auto *CE = dyn_cast<CStyleCastExpr>(UO->getSubExpr());
  if (!CE || !SemaBPF::isPreserveAIEnumeratorRef(CE->getSubExpr()))
    return;
  auto *DRE = cast<DeclRefExpr>(
      cast<PseudoObjectExpr>(CE->getSubExpr())->getSyntacticForm());
  CE->setSubExpr(DRE);
  CE->setCastKind(getEnumeratorToPointerCastKind(Context, DRE));
}

bool SemaBPF::CheckBPFBuiltinFunctionCall(unsigned BuiltinID,
                                          CallExpr *TheCall) {
  assert((BuiltinID == BPF::BI__builtin_preserve_field_info ||
          BuiltinID == BPF::BI__builtin_btf_type_id ||
          BuiltinID == BPF::BI__builtin_preserve_type_info ||
          BuiltinID == BPF::BI__builtin_preserve_enum_value) &&
         "unexpected BPF builtin");
  ASTContext &Context = getASTContext();
  if (SemaRef.checkArgCount(TheCall, 2))
    return true;

  // The second argument needs to be a constant int
  Expr *Arg = TheCall->getArg(1);
  std::optional<llvm::APSInt> Value = Arg->getIntegerConstantExpr(Context);
  diag::kind kind;
  if (!Value) {
    if (BuiltinID == BPF::BI__builtin_preserve_field_info)
      kind = diag::err_preserve_field_info_not_const;
    else if (BuiltinID == BPF::BI__builtin_btf_type_id)
      kind = diag::err_btf_type_id_not_const;
    else if (BuiltinID == BPF::BI__builtin_preserve_type_info)
      kind = diag::err_preserve_type_info_not_const;
    else
      kind = diag::err_preserve_enum_value_not_const;
    Diag(Arg->getBeginLoc(), kind) << 2 << Arg->getSourceRange();
    return true;
  }

  // The first argument
  Arg = TheCall->getArg(0);
  bool InvalidArg = false;
  bool ReturnUnsignedInt = true;
  if (BuiltinID == BPF::BI__builtin_preserve_field_info) {
    if (!isValidPreserveFieldInfoArg(Arg)) {
      InvalidArg = true;
      kind = diag::err_preserve_field_info_not_field;
    }
  } else if (BuiltinID == BPF::BI__builtin_preserve_type_info) {
    if (!isValidPreserveTypeInfoArg(Arg)) {
      InvalidArg = true;
      kind = diag::err_preserve_type_info_invalid;
    }
  } else if (BuiltinID == BPF::BI__builtin_preserve_enum_value) {
    unwrapPreserveEnumValueArg(Context, Arg);
    if (!isValidPreserveEnumValueArg(Arg)) {
      InvalidArg = true;
      kind = diag::err_preserve_enum_value_invalid;
    }
    ReturnUnsignedInt = false;
  } else if (BuiltinID == BPF::BI__builtin_btf_type_id) {
    ReturnUnsignedInt = false;
  }

  if (InvalidArg) {
    Diag(Arg->getBeginLoc(), kind) << 1 << Arg->getSourceRange();
    return true;
  }

  if (ReturnUnsignedInt)
    TheCall->setType(Context.UnsignedIntTy);
  else
    TheCall->setType(Context.UnsignedLongTy);
  return false;
}

void SemaBPF::handlePreserveAIRecord(RecordDecl *RD) {
  // Add preserve_access_index attribute to all fields and inner records.
  for (auto *D : RD->decls()) {
    if (D->hasAttr<BPFPreserveAccessIndexAttr>())
      continue;

    D->addAttr(BPFPreserveAccessIndexAttr::CreateImplicit(getASTContext()));
    if (auto *Rec = dyn_cast<RecordDecl>(D))
      handlePreserveAIRecord(Rec);
  }
}

void SemaBPF::handlePreserveAccessIndexAttr(Decl *D, const ParsedAttr &AL) {
  if (auto *Rec = dyn_cast<RecordDecl>(D))
    handlePreserveAIRecord(Rec);
  auto *A =
      ::new (getASTContext()) BPFPreserveAccessIndexAttr(getASTContext(), AL);
  if (auto *ED = dyn_cast<EnumDecl>(D);
      ED && !ED->getDeclName() && !AL.isPragmaClangAttribute())
    AnonPreserveAIEnums.insert({ED, A});
  D->addAttr(A);
}

void SemaBPF::DiagnoseAnonymousPreserveAIEnums() {
  for (auto [ED, A] : AnonPreserveAIEnums)
    if (!ED->getTypedefNameForAnonDecl())
      Diag(A->getLocation(), diag::warn_preserve_access_index_anon_enum) << A;
  AnonPreserveAIEnums.clear();
}

static QualType getPreserveAIEnumType(ASTContext &Context,
                                      const DeclRefExpr *E) {
  const auto *Enumerator = dyn_cast<EnumConstantDecl>(E->getDecl());
  if (!Enumerator)
    return QualType();

  // In C, an enumerator usually has type int rather than the enum type, so
  // find its enum through the declaration context. Within the definition of
  // the enum, e.g., A in B = A + 1, the enum is not complete yet and A must
  // stay a constant, so it is not relocated.
  const auto *ED = dyn_cast<EnumDecl>(Enumerator->getDeclContext());
  if (!ED || !ED->isCompleteDefinition() ||
      !ED->hasAttr<BPFPreserveAccessIndexAttr>())
    return QualType();

  // The relocation is matched by type name, so an anonymous enum is only
  // supported through its typedef name.
  if (ED->getDeclName())
    return Context.getCanonicalTagType(ED);
  if (const TypedefNameDecl *TD = ED->getTypedefNameForAnonDecl())
    return Context.getTypeDeclType(TD);
  return QualType();
}

Expr *SemaBPF::BuildPreserveAIEnumeratorRef(DeclRefExpr *E) {
  if (SemaRef.isUnevaluatedContext())
    return E;
  if (!SemaRef.hasReducedDebugInfo())
    return E;

  ASTContext &Context = getASTContext();
  QualType EnumTy = getPreserveAIEnumType(Context, E);
  if (EnumTy.isNull())
    return E;

  // Build *(enum_type *)E in the form required by
  // __builtin_preserve_enum_value().
  SourceLocation Loc = E->getExprLoc();
  auto *Ref = DeclRefExpr::Create(Context, E->getQualifierLoc(),
                                  SourceLocation(), E->getDecl(),
                                  /*RefersToEnclosingVariableOrCapture=*/false,
                                  E->getNameInfo(), E->getType(), VK_PRValue);
  QualType PtrTy = Context.getPointerType(EnumTy);
  Expr *Cast = CStyleCastExpr::Create(
      Context, PtrTy, VK_PRValue, getEnumeratorToPointerCastKind(Context, Ref),
      Ref,
      /*BasePath=*/nullptr, FPOptionsOverride(),
      Context.getTrivialTypeSourceInfo(PtrTy, Loc), Loc, Loc);
  Expr *Deref = UnaryOperator::Create(
      Context, Cast, UO_Deref, EnumTy, VK_LValue, OK_Ordinary, Loc,
      /*CanOverflow=*/false, FPOptionsOverride());

  // Flag 1 is BPF_ENUMVAL_VALUE.
  Expr *Flag =
      IntegerLiteral::Create(Context, llvm::APInt(32, 1), Context.IntTy, Loc);
  Expr *Args[] = {Deref, Flag};

  // Build the call the same way as the parser builds an explicit
  // __builtin_preserve_enum_value() call, so that the two are compatible, i.e.,
  // they have the same AST and are handled the same, e.g., with the same
  // diagnostics where a constant expression is required.
  // Sema::BuildBuiltinCallExpr() is not used since it refers to the builtin as
  // an ordinary function, for which the constant evaluator evaluates the
  // arguments and may give additional diagnostics.
  LookupResult R(SemaRef,
                 &Context.Idents.get(Context.BuiltinInfo.getName(
                     BPF::BI__builtin_preserve_enum_value)),
                 Loc, Sema::LookupOrdinaryName);
  SemaRef.LookupName(R, SemaRef.TUScope, /*AllowBuiltinCreation=*/true);
  auto *BuiltinDecl = R.getAsSingle<FunctionDecl>();
  assert(BuiltinDecl && "failed to find builtin declaration");
  ExprResult Callee = SemaRef.BuildDeclarationNameExpr(
      CXXScopeSpec(), R.getLookupNameInfo(), BuiltinDecl);
  assert(Callee.isUsable() && "Builtin reference cannot fail");
  ExprResult Call =
      SemaRef.BuildCallExpr(/*Scope=*/nullptr, Callee.get(), Loc, Args, Loc);
  assert(Call.isUsable() && "Call to builtin cannot fail");
  // The relocated value is not a constant, so the enumerator cannot be used
  // where a compile-time constant is required. Like an explicit builtin call,
  // it has type unsigned long rather than the type of the enumerator.
  return PseudoObjectExpr::Create(Context, E, {Call.get()}, /*resultIndex=*/0);
}

// Return whether E is an enumerator reference built by
// BuildPreserveAIEnumeratorRef().
bool SemaBPF::isPreserveAIEnumeratorRef(const Expr *E) {
  const auto *POE = dyn_cast<PseudoObjectExpr>(E);
  if (!POE)
    return false;
  const auto *DRE = dyn_cast<DeclRefExpr>(POE->getSyntacticForm());
  return DRE && isa<EnumConstantDecl>(DRE->getDecl());
}

} // namespace clang
