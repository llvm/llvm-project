//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseBracedInitializationCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

namespace {

AST_MATCHER_P(VarDecl, hasInitStyle, VarDecl::InitializationStyle, Style) {
  return Node.getInitStyle() == Style;
}

AST_MATCHER(Type, isDependentType) { return Node.isDependentType(); }

AST_MATCHER(CXXConstructExpr, noMacroParens) {
  const SourceRange Range = Node.getParenOrBraceRange();
  return Range.isValid() && !Range.getBegin().isMacroID() &&
         !Range.getEnd().isMacroID();
}

AST_MATCHER(Expr, isCXXParenListInitExpr) {
  return isa<CXXParenListInitExpr>(Node);
}

/// Matches CXXConstructExpr whose target class has any constructor
/// taking 'std::initializer_list'. When such a constructor exists, braced
/// initialization may call it instead of the intended constructor.
AST_MATCHER(CXXConstructExpr, constructsTypeWithInitListCtor) {
  const CXXRecordDecl *RD = Node.getConstructor()->getParent();
  if (!RD || !RD->hasDefinition())
    return false;
  return llvm::any_of(RD->ctors(), [](const CXXConstructorDecl *Ctor) {
    if (Ctor->getNumParams() == 0)
      return false;
    const QualType FirstParam =
        Ctor->getParamDecl(0)->getType().getNonReferenceType();
    const auto *Record = FirstParam->getAsCXXRecordDecl();
    if (!Record || !Record->getDeclName().isIdentifier() ||
        Record->getName() != "initializer_list" || !Record->isInStdNamespace())
      return false;
    // [dcl.init.list] p2: all other params must have defaults.
    for (unsigned I = 1; I < Ctor->getNumParams(); ++I)
      if (!Ctor->getParamDecl(I)->hasDefaultArg())
        return false;
    return true;
  });
}

} // namespace

void UseBracedInitializationCheck::registerMatchers(MatchFinder *Finder) {
  auto GoodCtor = allOf(
      noMacroParens(), unless(constructsTypeWithInitListCtor()),
      unless(isInTemplateInstantiation()), unless(isListInitialization()));
  auto GoodCtorExpr = cxxConstructExpr(GoodCtor).bind("ctor");
  auto GoodVar =
      allOf(unless(hasType(isDependentType())), unless(hasType(autoType())));

  // Variable declarations: Simple w(1), Takes t({1, 2})
  Finder->addMatcher(varDecl(hasInitStyle(VarDecl::CallInit),
                             hasInitializer(ignoringImplicit(GoodCtorExpr)),
                             GoodVar),
                     this);

  // Scalar direct-init: int x(42), double d(3.14)
  Finder->addMatcher(
      varDecl(hasInitStyle(VarDecl::CallInit),
              hasInitializer(unless(ignoringImplicit(cxxConstructExpr()))),
              GoodVar)
          .bind("scalar"),
      this);

  Finder->addMatcher(cxxFunctionalCastExpr(has(GoodCtorExpr)), this);
  Finder->addMatcher(cxxTemporaryObjectExpr(GoodCtor).bind("ctor"), this);
  Finder->addMatcher(cxxNewExpr(has(GoodCtorExpr)), this);

  if (getLangOpts().CPlusPlus20) {
    auto GoodPLE = expr(isCXXParenListInitExpr()).bind("ple");
    Finder->addMatcher(varDecl(hasInitStyle(VarDecl::ParenListInit),
                               hasInitializer(GoodPLE), GoodVar)
                           .bind("var_ple"),
                       this);
    Finder->addMatcher(cxxFunctionalCastExpr(has(GoodPLE)).bind("cast_ple"),
                       this);
    Finder->addMatcher(cxxNewExpr(has(GoodPLE)).bind("new_ple"), this);
  }
}

void UseBracedInitializationCheck::check(
    const MatchFinder::MatchResult &Result) {
  SourceLocation DiagLoc;
  SourceLocation LParen;
  SourceLocation RParen;

  if (const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructExpr>("ctor")) {
    DiagLoc = Ctor->getBeginLoc();
    LParen = Ctor->getParenOrBraceRange().getBegin();
    RParen = Ctor->getParenOrBraceRange().getEnd();
  } else if (const auto *Var = Result.Nodes.getNodeAs<VarDecl>("scalar")) {
    assert(Var->hasInit());
    const SourceManager &SM = *Result.SourceManager;
    const LangOptions &LangOpts = Result.Context->getLangOpts();

    const std::optional<Token> LTok =
        utils::lexer::findPreviousTokenSkippingComments(
            Var->getInit()->getBeginLoc(), SM, LangOpts);
    if (!LTok || LTok->isNot(tok::l_paren) || LTok->getLocation().isMacroID())
      return;

    const std::optional<Token> RTok =
        utils::lexer::findNextTokenSkippingComments(Var->getInit()->getEndLoc(),
                                                    SM, LangOpts);
    if (!RTok || RTok->isNot(tok::r_paren) || RTok->getLocation().isMacroID())
      return;

    DiagLoc = Var->getLocation();
    LParen = LTok->getLocation();
    RParen = RTok->getLocation();
  } else if (const auto *PLE =
                 Result.Nodes.getNodeAs<CXXParenListInitExpr>("ple")) {
    LParen = PLE->getBeginLoc();
    RParen = PLE->getEndLoc();
    if (const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var_ple"))
      DiagLoc = Var->getLocation();
    else if (const auto *Cast =
                 Result.Nodes.getNodeAs<CXXFunctionalCastExpr>("cast_ple"))
      DiagLoc = Cast->getBeginLoc();
    else if (const auto *New = Result.Nodes.getNodeAs<CXXNewExpr>("new_ple"))
      DiagLoc = New->getBeginLoc();
    else
      llvm_unreachable("No context for CXXParenListInitExpr");
  } else {
    llvm_unreachable("No matches found");
  }

  if (LParen.isMacroID() || RParen.isMacroID())
    return;

  diag(DiagLoc,
       "use braced initialization instead of parenthesized initialization")
      << FixItHint::CreateReplacement(LParen, "{")
      << FixItHint::CreateReplacement(RParen, "}");
}

} // namespace clang::tidy::misc
