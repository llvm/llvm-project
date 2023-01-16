//===--- NoexceptMoveConstructorCheck.cpp - clang-tidy---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoexceptMoveConstructorCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

void NoexceptMoveConstructorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxMethodDecl(anyOf(cxxConstructorDecl(), hasOverloadedOperatorName("=")),
                    unless(isImplicit()), unless(isDeleted()))
          .bind("decl"),
      this);
}

void NoexceptMoveConstructorCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *Decl = Result.Nodes.getNodeAs<CXXMethodDecl>("decl")) {
    bool IsConstructor = false;
    if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(Decl)) {
      if (!Ctor->isMoveConstructor())
        return;
      IsConstructor = true;
    } else if (!Decl->isMoveAssignmentOperator()) {
      return;
    }

    const auto *ProtoType = Decl->getType()->castAs<FunctionProtoType>();

    if (isUnresolvedExceptionSpec(ProtoType->getExceptionSpecType()))
      return;

    if (!isNoexceptExceptionSpec(ProtoType->getExceptionSpecType())) {
      auto Diag = diag(Decl->getLocation(),
                       "move %select{assignment operator|constructor}0s should "
                       "be marked noexcept")
                  << IsConstructor;
      // Add FixIt hints.
      SourceManager &SM = *Result.SourceManager;
      assert(Decl->getNumParams() > 0);
      SourceLocation NoexceptLoc = Decl->getParamDecl(Decl->getNumParams() - 1)
                                       ->getSourceRange()
                                       .getEnd();
      if (NoexceptLoc.isValid())
        NoexceptLoc = Lexer::findLocationAfterToken(
            NoexceptLoc, tok::r_paren, SM, Result.Context->getLangOpts(), true);
      if (NoexceptLoc.isValid())
        Diag << FixItHint::CreateInsertion(NoexceptLoc, " noexcept ");
      return;
    }

    // Don't complain about nothrow(false), but complain on nothrow(expr)
    // where expr evaluates to false.
    if (ProtoType->canThrow() == CT_Can) {
      Expr *E = ProtoType->getNoexceptExpr();
      E = E->IgnoreImplicit();
      if (!isa<CXXBoolLiteralExpr>(E)) {
        diag(E->getExprLoc(),
             "noexcept specifier on the move %select{assignment "
             "operator|constructor}0 evaluates to 'false'")
            << IsConstructor;
      }
    }
  }
}

} // namespace clang::tidy::performance
