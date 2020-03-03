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
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace performance {

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
    StringRef MethodType = "assignment operator";
    if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(Decl)) {
      if (!Ctor->isMoveConstructor())
        return;
      MethodType = "constructor";
    } else if (!Decl->isMoveAssignmentOperator()) {
      return;
    }

    const auto *ProtoType = Decl->getType()->getAs<FunctionProtoType>();

    if (isUnresolvedExceptionSpec(ProtoType->getExceptionSpecType()))
      return;

    if (!isNoexceptExceptionSpec(ProtoType->getExceptionSpecType())) {
      auto Diag =
          diag(Decl->getLocation(), "move %0s should be marked noexcept")
          << MethodType;
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
             "noexcept specifier on the move %0 evaluates to 'false'")
            << MethodType;
      }
    }
  }
}

} // namespace performance
} // namespace tidy
} // namespace clang
