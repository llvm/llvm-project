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

// Manual parent walk rather than a matcher because implicit
// wrappers obscure assignment contexts.
static const VarDecl *findParentVarDecl(ASTContext &Ctx, const Stmt *S) {
  DynTypedNode Node = DynTypedNode::create(*S);

  for (;;) {
    auto Parents = Ctx.getParents(Node);
    if (Parents.empty())
      return nullptr;

    const Expr *NextExpr = nullptr;
    const Stmt *NextStmt = nullptr;

    for (const DynTypedNode &P : Parents) {
      if (const auto *VD = P.get<VarDecl>())
        return VD;

      // Exprs first preserves semantic structure longer than bare Stmt.
      if (!NextExpr)
        NextExpr = P.get<Expr>();
      if (!NextStmt)
        NextStmt = P.get<Stmt>();
    }

    if (NextExpr)
      Node = DynTypedNode::create(*NextExpr);
    else if (NextStmt)
      Node = DynTypedNode::create(*NextStmt);
    else
      return nullptr;
  }
}

// Returns a StringRef into the SourceManager-owned buffer; stable for lifetime
// of the ASTContext.
static StringRef extractWrittenElementType(const TypeSourceInfo *TSI,
                                           SourceManager &SM,
                                           const LangOptions &LO) {
  if (!TSI)
    return {};
  const TypeLoc TL = TSI->getTypeLoc().getUnqualifiedLoc();
  auto TSTL = TL.getAsAdjusted<TemplateSpecializationTypeLoc>();
  if (!TSTL || TSTL.getNumArgs() < 1)
    return {};
  const TypeSourceInfo *ArgTSI = TSTL.getArgLoc(0).getTypeSourceInfo();
  if (!ArgTSI)
    return {};
  const TypeLoc ArgTL = ArgTSI->getTypeLoc();
  const CharSourceRange R =
      CharSourceRange::getTokenRange(ArgTL.getBeginLoc(), ArgTL.getEndLoc());
  return Lexer::getSourceText(R, SM, LO);
}

// Returns the QualType of the deleter's pointee, or null if the
// deleter shape is not recognised.
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

// Locates the closing '>' of the shared_ptr<T> template-id, RAngleLoc is the
// '>' to patch in the constructor expression. Locates separately the '>' of the
// VarDecl's declared type when different (copy-init, pointer/ref declarator).
static void resolveRAngleLocs(const CXXConstructExpr *Ctor,
                              const VarDecl *ParentVD, SourceManager &SM,
                              const LangOptions &LO, SourceLocation &RAngleLoc,
                              SourceLocation &VDRAngleLoc) {
  auto FindRAngleLoc = [&](const TypeSourceInfo *TSI) -> SourceLocation {
    if (!TSI)
      return {};
    TypeLoc TL = TSI->getTypeLoc();
    if (auto QTL = TL.getAs<QualifiedTypeLoc>())
      TL = QTL.getUnqualifiedLoc();
    if (auto TSTL = TL.getAsAdjusted<TemplateSpecializationTypeLoc>())
      return TSTL.getRAngleLoc();
    return {};
  };

  if (ParentVD) {
    // Pointer/reference declarators (e.g. shared_ptr<A> *sp = ...) wrap the
    // template-id in a way that prevents safely rewriting both the declared
    // type and constructor expression independently. Warn only.
    const TypeLoc TL = ParentVD->getTypeSourceInfo()->getTypeLoc();
    if (!TL.getAs<PointerTypeLoc>().isNull() ||
        !TL.getAs<ReferenceTypeLoc>().isNull()) {
      RAngleLoc = {};
      VDRAngleLoc = {};
      return;
    }

    RAngleLoc = FindRAngleLoc(ParentVD->getTypeSourceInfo());

    if (RAngleLoc.isInvalid()) {
      // Auto-declared VarDecl carry no written template-id in TypeSourceInfo.
      // Try constructor TSI first, then fall back to token scanning between the
      // constructor start and its paren/brace range.
      SourceLocation CtorRAngleLoc;

      if (const auto *TOE = dyn_cast<CXXTemporaryObjectExpr>(Ctor))
        CtorRAngleLoc = FindRAngleLoc(TOE->getTypeSourceInfo());

      if (CtorRAngleLoc.isInvalid()) {
        const SourceLocation CtorLoc = Ctor->getBeginLoc();
        const SourceLocation LParenLoc =
            Ctor->getParenOrBraceRange().getBegin();

        if (CtorLoc.isValid() && LParenLoc.isValid() && !CtorLoc.isMacroID() &&
            !LParenLoc.isMacroID()) {
          SourceLocation ScanLoc = CtorLoc;

          // Fallback: when the VarDecl is declared with 'auto', the
          // TypeSourceInfo carries no written template-id; scan tokens between
          // the constructor's start and its paren/brace to locate the last '>'
          // before the argument list.
          for (;;) {
            std::optional<Token> MaybeTok =
                Lexer::findNextToken(ScanLoc, SM, LO);
            if (!MaybeTok)
              break;

            const Token &T = *MaybeTok;

            if (T.is(tok::l_paren) || T.is(tok::l_brace))
              break;

            if (T.is(tok::greater) || T.is(tok::greatergreater))
              CtorRAngleLoc = T.getLocation();

            const SourceLocation Next =
                Lexer::getLocForEndOfToken(T.getLocation(), 0, SM, LO);

            // Guard against Lexer::getLocForEndOfToken returning the same
            // location on malformed tokens/infinite loop.
            if (Next == ScanLoc)
              break;

            ScanLoc = Next;
          }
        }
      }

      RAngleLoc = CtorRAngleLoc;

    } else if (ParentVD->getInitStyle() == VarDecl::CInit) {
      // Copy-init: VarDecl and constructor expression have separate
      // template-ids in source, both require independent insertions.
      VDRAngleLoc = RAngleLoc;

      if (const auto *TOE = dyn_cast<CXXTemporaryObjectExpr>(Ctor))
        RAngleLoc = FindRAngleLoc(TOE->getTypeSourceInfo());

      // Inserting twice into the same token would produce T[][].
      if (RAngleLoc.isInvalid() || RAngleLoc == VDRAngleLoc)
        RAngleLoc = {};
    }

  } else if (const auto *TOE = dyn_cast<CXXTemporaryObjectExpr>(Ctor)) {
    // No VarDecl parent: standalone temporary or return expression.
    RAngleLoc = FindRAngleLoc(TOE->getTypeSourceInfo());
  }
}

// Manual parent walk rather than a matcher because implicit
// wrappers obscure assignment contexts.
static bool isInsideAssignment(ASTContext &Ctx, const CXXConstructExpr *Ctor) {
  DynTypedNode Node = DynTypedNode::create(*Ctor);
  for (;;) {
    auto Parents = Ctx.getParents(Node);
    if (Parents.empty())
      return false;

    bool Advanced = false;
    for (const DynTypedNode &P : Parents) {
      if (const auto *Op = P.get<CXXOperatorCallExpr>())
        if (Op->getOperator() == OO_Equal)
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

// FixIt 1: insert "[]" after the closing '>' of the shared_ptr<T> template-id.
// ">>" tokens (merged with an enclosing template's '>') get a +1 offset to get
// ">[]>" rather than "[]>>".
static FixItHint makeArrayInsertionFix(SourceLocation Loc, SourceManager &SM,
                                       const LangOptions &LO) {
  Token AngleTok;
  if (!Lexer::getRawToken(Loc, AngleTok, SM, LO))
    if (AngleTok.is(tok::greatergreater))
      Loc = Loc.getLocWithOffset(1);

  return FixItHint::CreateInsertion(Loc, "[]");
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
  const VarDecl *ParentVD = findParentVarDecl(Ctx, Ctor);
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
    const auto &VDParents = Ctx.getParents(*ParentVD);
    if (!VDParents.empty())
      if (const auto *DS = VDParents[0].get<DeclStmt>())
        if (!DS->isSingleDecl()) {
          Warn();
          return;
        }
  }

  // Assignment targets may not have a rewritable written type at the
  // declaration site. Warn only.
  if (isInsideAssignment(Ctx, Ctor)) {
    Warn();
    return;
  }

  SourceLocation RAngleLoc;
  SourceLocation VDRAngleLoc;
  resolveRAngleLocs(Ctor, ParentVD, SM, LO, RAngleLoc, VDRAngleLoc);

  // FixIt 1: Insert [] into the rewritten shared_ptr type.
  FixItHint InsertArrayVarDecl;
  if (VDRAngleLoc.isValid() && !VDRAngleLoc.isMacroID())
    InsertArrayVarDecl = makeArrayInsertionFix(VDRAngleLoc, SM, LO);

  // FixIt 2: Remove the explicit deleter argument.
  auto RemoveDeleter = buildRemoveDeleterFix(Ctor, SM, LO);
  if (!RemoveDeleter)
    return;

  auto Diag = Warn();
  if (RAngleLoc.isInvalid() || RAngleLoc.isMacroID())
    return;

  Diag << makeArrayInsertionFix(RAngleLoc, SM, LO) << *RemoveDeleter
       << InsertArrayVarDecl;
}

} // namespace clang::tidy::modernize
