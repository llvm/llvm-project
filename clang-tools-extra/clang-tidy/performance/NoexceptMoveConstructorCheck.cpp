//===--- NoexceptMoveConstructorCheck.cpp - clang-tidy---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoexceptMoveConstructorCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

void NoexceptMoveConstructorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxMethodDecl(unless(isDeleted()),
                    anyOf(cxxConstructorDecl(isMoveConstructor()),
                          isMoveAssignmentOperator()))
          .bind("decl"),
      this);
}

void NoexceptMoveConstructorCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *FuncDecl = Result.Nodes.getNodeAs<CXXMethodDecl>("decl");
  assert(FuncDecl);

  if (SpecAnalyzer.analyze(FuncDecl) !=
      utils::ExceptionSpecAnalyzer::State::Throwing)
    return;

  const bool IsConstructor = CXXConstructorDecl::classof(FuncDecl);

  // Don't complain about nothrow(false), but complain on nothrow(expr)
  // where expr evaluates to false.
  const auto *ProtoType = FuncDecl->getType()->castAs<FunctionProtoType>();
  const Expr *NoexceptExpr = ProtoType->getNoexceptExpr();
  if (NoexceptExpr) {
    NoexceptExpr = NoexceptExpr->IgnoreImplicit();
    if (!isa<CXXBoolLiteralExpr>(NoexceptExpr)) {
      diag(NoexceptExpr->getExprLoc(),
           "noexcept specifier on the move %select{assignment "
           "operator|constructor}0 evaluates to 'false'")
          << IsConstructor;
    }
    return;
  }

  auto Diag = diag(FuncDecl->getLocation(),
                   "move %select{assignment operator|constructor}0s should "
                   "be marked noexcept")
              << IsConstructor;
  // Add FixIt hints.

  const SourceManager &SM = *Result.SourceManager;

  const SourceLocation NoexceptLoc =
      utils::lexer::getLocationForNoexceptSpecifier(FuncDecl, SM);
  if (NoexceptLoc.isValid())
    Diag << FixItHint::CreateInsertion(NoexceptLoc, " noexcept ");
}

} // namespace clang::tidy::performance
