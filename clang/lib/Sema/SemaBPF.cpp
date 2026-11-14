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
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
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

// In __builtin_preserve_enum_value(*(<enum_type> *)<enumerator>, flag), the
// enumerator names itself, not its value, so undo
// BuildPreserveAIEnumeratorRef() there, as if there were no attribute.
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
  // Add preserve_access_index attribute to all fields and inner records, but
  // not inner enums, so that their enumerators stay constants.
  for (auto *D : RD->decls()) {
    if (isa<EnumDecl>(D) || D->hasAttr<BPFPreserveAccessIndexAttr>())
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

// Return the enum type, or its typedef name, that the enumerator E is relocated
// against, or a null type if E is not relocated.
static QualType getPreserveAIEnumType(ASTContext &Context,
                                      const DeclRefExpr *E) {
  const auto *Enumerator = dyn_cast<EnumConstantDecl>(E->getDecl());
  if (!Enumerator)
    return QualType();

  // In C, an enumerator usually has type int, not the enum type, so find the
  // enum from its context. Within the enum's definition, e.g., A in B = A + 1,
  // the enum is incomplete and A stays a constant.
  const auto *ED = dyn_cast<EnumDecl>(Enumerator->getDeclContext());
  if (!ED || !ED->isCompleteDefinition() ||
      !ED->hasAttr<BPFPreserveAccessIndexAttr>())
    return QualType();

  // libbpf matches relocations by type name, so an anonymous enum needs a
  // typedef name.
  if (ED->getDeclName())
    return Context.getCanonicalTagType(ED);
  if (const TypedefNameDecl *TD = ED->getTypedefNameForAnonDecl())
    return Context.getTypeDeclType(TD);
  return QualType();
}

// Return the enumerator reference E, or, if it is relocated, a pseudo-object
// whose semantic form is a __builtin_preserve_enum_value() call.
Expr *SemaBPF::BuildPreserveAIEnumeratorRef(DeclRefExpr *E) {
  if (SemaRef.isUnevaluatedContext())
    return E;

  ASTContext &Context = getASTContext();
  QualType EnumTy = getPreserveAIEnumType(Context, E);
  if (EnumTy.isNull())
    return E;

  // The value cannot be relocated where a constant is required: in a constant
  // evaluated context, e.g., an array bound or a case label, or outside of a
  // function, e.g., a global initializer. Warn, and use the enumerator's value.
  if (SemaRef.isConstantEvaluatedContext() ||
      SemaRef.CurContext->getRedeclContext()->isFileContext()) {
    Diag(E->getExprLoc(),
         diag::warn_preserve_access_index_enumerator_not_relocated)
        << E->getDecl() << EnumTy << E->getSourceRange();
    return E;
  }

  // Keep it a constant anywhere in an unevaluated operand, e.g., in a block
  // literal in sizeof. TransformToPotentiallyEvaluated() relocates it when it
  // rebuilds a variably modified operand.
  if (llvm::any_of(SemaRef.ExprEvalContexts,
                   [](const Sema::ExpressionEvaluationContextRecord &Rec) {
                     return Rec.isUnevaluated();
                   }))
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
  Expr *Conv = SemaRef.BuildBuiltinCallExpr(
      Loc, BPF::BI__builtin_preserve_enum_value, Args);

  // Cast the unsigned long result to the type of the enumerator, normally int,
  // so that no type changes, e.g., hash % A stays a 32-bit unsigned remainder.
  QualType ResultTy = E->getType();
  if (!Context.hasSameType(Conv->getType(), ResultTy))
    Conv = CStyleCastExpr::Create(
        Context, ResultTy, VK_PRValue, CK_IntegralCast, Conv,
        /*BasePath=*/nullptr, FPOptionsOverride(),
        Context.getTrivialTypeSourceInfo(ResultTy, Loc), Loc, Loc);

  // E is the syntactic form, so the AST is printed as written.
  return PseudoObjectExpr::Create(Context, E, {Conv}, /*resultIndex=*/0);
}

// Return whether E is an enumerator reference built by
// BuildPreserveAIEnumeratorRef(), i.e., a pseudo-object for an enumerator whose
// semantic form is a __builtin_preserve_enum_value() call.
bool SemaBPF::isPreserveAIEnumeratorRef(const Expr *E) {
  const auto *POE = dyn_cast<PseudoObjectExpr>(E);
  if (!POE)
    return false;
  const auto *DRE = dyn_cast<DeclRefExpr>(POE->getSyntacticForm());
  if (!DRE || !isa<EnumConstantDecl>(DRE->getDecl()))
    return false;
  const auto *Call = dyn_cast<CallExpr>(POE->getResultExpr()->IgnoreCasts());
  return Call &&
         Call->getBuiltinCallee() == BPF::BI__builtin_preserve_enum_value;
}

// If a relocated enumerator makes E non-constant, return the constant
// evaluator's note saying so.
static std::optional<PartialDiagnosticAt>
getPreserveAIEnumeratorNote(ASTContext &Context, const Expr *E) {
  if (!Context.getTargetInfo().getTriple().isBPF() || E->isValueDependent())
    return std::nullopt;
  SmallVector<PartialDiagnosticAt, 8> Notes;
  Expr::EvalResult EvalResult;
  EvalResult.Diag = &Notes;
  if (E->EvaluateAsRValue(EvalResult, Context, /*InConstantContext=*/true))
    return std::nullopt;
  for (PartialDiagnosticAt &Note : Notes)
    if (Note.second.getDiagID() ==
        diag::note_constexpr_bpf_preserve_ai_enumerator)
      return std::move(Note);
  return std::nullopt;
}

// After an error that E is not a constant, emit the note if a relocated
// enumerator is the reason.
void SemaBPF::notePreserveAIEnumerator(const Expr *E) {
  if (std::optional<PartialDiagnosticAt> Note =
          getPreserveAIEnumeratorNote(getASTContext(), E))
    Diag(Note->first, Note->second);
}

// Replace each relocated enumerator in St with its DeclRefExpr, which has the
// same type and value kind, and warn that it is not relocated.
static Stmt *keepEnumeratorsConstant(SemaBPF &S, Stmt *St) {
  if (!St)
    return St;
  if (auto *E = dyn_cast<Expr>(St);
      E && SemaBPF::isPreserveAIEnumeratorRef(E)) {
    auto *DRE =
        cast<DeclRefExpr>(cast<PseudoObjectExpr>(E)->getSyntacticForm());
    S.Diag(DRE->getExprLoc(),
           diag::warn_preserve_access_index_enumerator_not_relocated)
        << DRE->getDecl() << getPreserveAIEnumType(S.getASTContext(), DRE)
        << DRE->getSourceRange();
    return DRE;
  }
  for (Stmt *&Child : St->children())
    Child = keepEnumeratorsConstant(S, Child);
  return St;
}

// E must be a constant, e.g., an array designator, so keep its enumerators
// constants and warn.
Expr *SemaBPF::keepPreserveAIEnumeratorsConstant(Expr *E) {
  return cast<Expr>(keepEnumeratorsConstant(*this, E));
}

// An asm operand E such as "i" must be a constant. If a relocated enumerator
// makes it non-constant, keep its enumerators constants and warn.
Expr *SemaBPF::checkAsmImmediateOperand(Expr *E) {
  if (!getPreserveAIEnumeratorNote(getASTContext(), E))
    return E;
  return keepPreserveAIEnumeratorsConstant(E);
}

// The initializer of a static or constexpr local variable must be a constant,
// so keep its enumerators constants and warn.
Expr *SemaBPF::checkLocalConstantInitializer(const VarDecl *VD, Expr *Init) {
  if (!VD->isLocalVarDecl() || (!VD->isStaticLocal() && !VD->isConstexpr()))
    return Init;
  return keepPreserveAIEnumeratorsConstant(Init);
}

} // namespace clang
