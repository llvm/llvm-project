//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseSharedPtrArrayCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

AST_MATCHER(FunctionDecl, funcHasSingleArrayDeleteBody) {
  if (Node.getNumParams() != 1 || !Node.hasBody())
    return false;
  const ParmVarDecl *Param = Node.getParamDecl(0);
  const auto *CS = dyn_cast<CompoundStmt>(Node.getBody());
  if (!CS || CS->size() != 1)
    return false;
  const auto *E = dyn_cast<Expr>(CS->body_front());
  if (!E)
    return false;
  const auto *DE = dyn_cast<CXXDeleteExpr>(E->IgnoreParenImpCasts());
  if (!DE || !DE->isArrayForm())
    return false;
  const auto *DRE =
      dyn_cast<DeclRefExpr>(DE->getArgument()->IgnoreParenImpCasts());
  return DRE && DRE->getDecl() == Param;
}

AST_MATCHER(LambdaExpr, lambdaHasSingleArrayDeleteBody) {
  if (Node.capture_size() != 0)
    return false;
  const CXXMethodDecl *CallOp = Node.getCallOperator();
  if (!CallOp || CallOp->getNumParams() != 1)
    return false;
  const ParmVarDecl *Param = CallOp->getParamDecl(0);
  const auto *CS = dyn_cast<CompoundStmt>(Node.getBody());
  if (!CS || CS->size() != 1)
    return false;
  const auto *E = dyn_cast<Expr>(CS->body_front());
  if (!E)
    return false;
  const auto *DE = dyn_cast<CXXDeleteExpr>(E->IgnoreParenImpCasts());
  if (!DE || !DE->isArrayForm())
    return false;
  const auto *DRE =
      dyn_cast<DeclRefExpr>(DE->getArgument()->IgnoreParenImpCasts());
  return DRE && DRE->getDecl() == Param;
}

void UseSharedPtrArrayCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxConstructExpr(
          unless(isInTemplateInstantiation()),
          hasType(qualType(hasDeclaration(classTemplateSpecializationDecl(
              hasName("::std::shared_ptr"), templateArgumentCountIs(1))))),

          argumentCountIs(2),

          hasArgument(
              0, ignoringParenImpCasts(cxxNewExpr(isArray()).bind("newExpr"))),

          hasArgument(
              1, ignoringImplicit(anyOf(

                     cxxConstructExpr(
                         hasType(qualType(hasCanonicalType(
                             hasDeclaration(classTemplateSpecializationDecl(
                                 hasName("::std::default_delete")))))))
                         .bind("defaultDelete"),

                     lambdaExpr(lambdaHasSingleArrayDeleteBody())
                         .bind("lambdaDeleter"),

                     declRefExpr(to(functionDecl(funcHasSingleArrayDeleteBody())
                                        .bind("deleterFunction")))))))

          .bind("sharedPtrCtor"),
      this);
}

// From bugprone-smart-ptr-array-mismatch-check
// Same as SmartPtrArrayMismatchCheck::getConstructedVarOrField.
static const DeclaratorDecl *
getConstructedVarOrField(const Expr *FoundConstructExpr, ASTContext &Ctx) {
  const DynTypedNodeList ConstructParents =
      Ctx.getParentMapContext().getParents(*FoundConstructExpr);
  if (ConstructParents.size() != 1)
    return nullptr;
  const auto *ParentDecl = ConstructParents.begin()->get<DeclaratorDecl>();
  if (isa_and_nonnull<VarDecl, FieldDecl>(ParentDecl))
    return ParentDecl;

  return nullptr;
}

// Returns a StringRef into the SourceManager-owned buffer; stable for lifetime
// of the ASTContext.
static StringRef extractWrittenElementType(const TypeSourceInfo *TSI,
                                           SourceManager &SM,
                                           const LangOptions &LO) {
  if (!TSI)
    return {};
  const TypeLoc TL = TSI->getTypeLoc().getUnqualifiedLoc();
  auto TSTypeLoc = TL.getAsAdjusted<TemplateSpecializationTypeLoc>();
  if (!TSTypeLoc || TSTypeLoc.getNumArgs() < 1)
    return {};
  const TypeSourceInfo *ArgTSI = TSTypeLoc.getArgLoc(0).getTypeSourceInfo();
  if (!ArgTSI)
    return {};
  const TypeLoc ArgTL = ArgTSI->getTypeLoc();
  const CharSourceRange R =
      CharSourceRange::getTokenRange(ArgTL.getBeginLoc(), ArgTL.getEndLoc());
  return Lexer::getSourceText(R, SM, LO);
}

static QualType getDeleterParamPointee(const Expr *DeleterArg) {
  const Expr *E = DeleterArg->IgnoreParenImpCasts();

  if (const auto *L = dyn_cast<LambdaExpr>(E)) {
    const CXXMethodDecl *CallOp = L->getCallOperator();
    if (!CallOp || CallOp->getNumParams() != 1)
      return {};
    const QualType P = CallOp->getParamDecl(0)->getType();
    if (!P->isPointerType())
      return {};
    return P->getPointeeType();
  }

  if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (const auto *FD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
      if (FD->getNumParams() != 1)
        return {};
      const QualType P = FD->getParamDecl(0)->getType();
      if (!P->isPointerType())
        return {};
      return P->getPointeeType();
    }
  }

  // default_delete<T[]>: the template argument is T[], extract T.
  if (const auto *CE = dyn_cast<CXXConstructExpr>(E)) {
    const auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(
        CE->getConstructor()->getParent());
    if (!CTSD || CTSD->getTemplateArgs().size() != 1)
      return {};
    const TemplateArgument &Arg = CTSD->getTemplateArgs()[0];
    if (Arg.getKind() != TemplateArgument::Type)
      return {};
    const QualType T = Arg.getAsType();
    if (!T->isArrayType())
      return {};
    return cast<ArrayType>(T.getTypePtr())->getElementType();
  }

  return {};
}

// Manual parent walk rather than a matcher because implicit
// wrappers obscure assignment contexts.
static bool isInsideAssignment(ASTContext &Ctx, const CXXConstructExpr *Ctor) {
  DynTypedNode Node = DynTypedNode::create(*Ctor);
  while (true) {
    auto Parents = Ctx.getParents(Node);
    if (Parents.empty())
      return false;

    bool Advanced = false;
    for (const DynTypedNode &P : Parents) {
      if (const auto *Op = P.get<CXXOperatorCallExpr>();
          Op && Op->getOperator() == OO_Equal)
        return true;

      // VarDecl indicates initialization rather than assignment.
      if (P.get<VarDecl>())
        return false;

      if (!Advanced && (P.get<Expr>() || P.get<CXXBindTemporaryExpr>() ||
                        P.get<MaterializeTemporaryExpr>())) {
        Node = P;
        Advanced = true;
      }
    }

    if (!Advanced)
      return false;
  }
}

// FixIt 2: remove ", deleter" — from the end of arg 0 to the end of arg 1.
static std::optional<FixItHint>
buildRemoveDeleterFix(const CXXConstructExpr *Ctor, SourceManager &SM,
                      const LangOptions &LO) {
  const Expr *Arg0 = Ctor->getArg(0);
  const Expr *Arg1 = Ctor->getArg(1);

  const SourceLocation Arg0End =
      Lexer::getLocForEndOfToken(Arg0->getEndLoc(), 0, SM, LO);
  const SourceLocation Arg1End =
      Lexer::getLocForEndOfToken(Arg1->getEndLoc(), 0, SM, LO);

  if (Arg0End.isInvalid() || Arg1End.isInvalid())
    return std::nullopt;

  return FixItHint::CreateRemoval(
      CharSourceRange::getCharRange(Arg0End, Arg1End));
}

void UseSharedPtrArrayCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructExpr>("sharedPtrCtor");
  assert(Ctor && "sharedPtrCtor must be bound");

  const auto *NewExpr = Result.Nodes.getNodeAs<CXXNewExpr>("newExpr");
  assert(NewExpr && "newExpr must be bound");

  const auto *CTSD = cast<ClassTemplateSpecializationDecl>(
      Ctor->getType()->getAsCXXRecordDecl());
  assert(CTSD->getTemplateArgs().size() == 1 &&
         "shared_ptr must have exactly one template argument");

  const TemplateArgument &TyArg = CTSD->getTemplateArgs()[0];
  assert(TyArg.getKind() == TemplateArgument::Type &&
         "shared_ptr template argument must be a type");

  QualType ElemTy = TyArg.getAsType();
  if (ElemTy->isArrayType() || ElemTy->isDependentType())
    return;

  QualType AllocTy = NewExpr->getAllocatedType();
  if (AllocTy.isNull())
    return;
  if (AllocTy->isArrayType())
    AllocTy = cast<ArrayType>(AllocTy.getTypePtr())->getElementType();

  ASTContext &Ctx = *Result.Context;

  // Unqualified canonical comparison: handles CV qualified arrays and typedefs.
  const QualType SharedElem = ElemTy.getCanonicalType().getUnqualifiedType();
  if (!clang::ASTContext::hasSameType(
          AllocTy.getCanonicalType().getUnqualifiedType(), SharedElem))
    return;

  const QualType DelPointee = getDeleterParamPointee(Ctor->getArg(1));
  if (DelPointee.isNull())
    return;
  if (!clang::ASTContext::hasSameType(
          DelPointee.getCanonicalType().getUnqualifiedType(), SharedElem))
    return;

  if (Ctor->getArg(0)->getBeginLoc().isMacroID() ||
      Ctor->getArg(1)->getEndLoc().isMacroID())
    return;

  SourceManager &SM = *Result.SourceManager;
  const LangOptions &LO = Ctx.getLangOpts();

  // CanonicalFallbackBuff anchors the StringRef only in the canonical fallback;
  // extractWrittenElementType returns into SourceManager owned memory so needs
  // no buffer.
  const DeclaratorDecl *ParentVD = getConstructedVarOrField(Ctor, Ctx);
  std::string CanonicalFallbackBuff;
  StringRef WrittenType = [&]() -> StringRef {
    if (ParentVD)
      if (StringRef S =
              extractWrittenElementType(ParentVD->getTypeSourceInfo(), SM, LO);
          !S.empty())
        return S;
    if (const auto *TOE = dyn_cast<CXXTemporaryObjectExpr>(Ctor))
      if (StringRef S =
              extractWrittenElementType(TOE->getTypeSourceInfo(), SM, LO);
          !S.empty())
        return S;
    CanonicalFallbackBuff = ElemTy.getAsString(Ctx.getPrintingPolicy());
    return CanonicalFallbackBuff;
  }();

  auto Warn = [&]() -> DiagnosticBuilder {
    return diag(Ctor->getBeginLoc(),
                "use 'std::shared_ptr<%0[]>' instead of "
                "'std::shared_ptr<%0>' with explicit array deleter")
           << WrittenType.str();
  };

  // Multi-declarator: one TypeLoc shared across all declarators. Warn only.
  if (ParentVD) {
    const TypeLoc TL = ParentVD->getTypeSourceInfo()->getTypeLoc();
    if (!TL.getAs<PointerTypeLoc>().isNull() ||
        !TL.getAs<ReferenceTypeLoc>().isNull()) {
      Warn();
      return;
    }
    if (const auto &VDParents = Ctx.getParents(*ParentVD); !VDParents.empty())
      if (const auto *DS = VDParents[0].get<DeclStmt>();
          DS && !DS->isSingleDecl()) {
        Warn();
        return;
      }
    if (const auto *VD = dyn_cast<VarDecl>(ParentVD)) {
      const Expr *Init = VD->getInit();
      if (Init && Init->IgnoreImplicit() != Ctor) {
        Warn();
        return;
      }
    }
  }

  // Assignment targets may not have a rewritable written type at the
  // declaration site. Warn only.
  if (isInsideAssignment(Ctx, Ctor)) {
    Warn();
    return;
  }

  // FixIt 1: Insert [] into the rewritten shared_ptr type.
  //  Same inline logic as bugprone-smart-ptr-array-check.
  auto GetInsertLoc = [&](const TypeSourceInfo *TSI) -> SourceLocation {
    if (!TSI)
      return {};
    auto TSTypeLoc =
        TSI->getTypeLoc().getAsAdjusted<TemplateSpecializationTypeLoc>();
    if (!TSTypeLoc || TSTypeLoc.getNumArgs() < 1)
      return {};

    const TypeSourceInfo *ArgTSI = TSTypeLoc.getArgLoc(0).getTypeSourceInfo();

    const SourceRange TemplateArgumentRange =
        ArgTSI->getTypeLoc().getSourceRange();

    return Lexer::getLocForEndOfToken(TemplateArgumentRange.getEnd(), 0, SM,
                                      LO);
  };

  const TypeSourceInfo *VdTSI =
      ParentVD ? ParentVD->getTypeSourceInfo() : nullptr;
  const TypeSourceInfo *CtorTSI = nullptr;

  if (const auto *TOE = dyn_cast<CXXTemporaryObjectExpr>(Ctor))
    CtorTSI = TOE->getTypeSourceInfo();
  if (!ParentVD && CtorTSI) {
    const DynTypedNodeList CtorParents =
        Ctx.getParentMapContext().getParents(*Ctor);
    if (CtorParents.size() == 1 && CtorParents.begin()->get<Expr>()) {
      Warn();
      return;
    }
  }

  const SourceLocation CtorInsertLoc = GetInsertLoc(CtorTSI ? CtorTSI : VdTSI);
  if (CtorInsertLoc.isInvalid()) {
    Warn();
    return;
  }

  auto RemoveDeleter = buildRemoveDeleterFix(Ctor, SM, LO);
  if (!RemoveDeleter)
    return;

  auto Diag = Warn();
  Diag << FixItHint::CreateInsertion(CtorInsertLoc, "[]") << *RemoveDeleter;

  if (VdTSI && CtorTSI) {
    const SourceLocation VDInsertLoc = GetInsertLoc(VdTSI);
    if (VDInsertLoc.isValid() && VDInsertLoc != CtorInsertLoc)
      Diag << FixItHint::CreateInsertion(VDInsertLoc, "[]");
  }
}

} // namespace clang::tidy::modernize
