//===--- ElseAfterReturnCheck.cpp - clang-tidy-----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ElseAfterReturnCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

void ElseAfterReturnCheck::registerMatchers(MatchFinder *Finder) {
  const auto ControlFlowInterruptorMatcher =
      stmt(anyOf(returnStmt().bind("return"), continueStmt().bind("continue"),
                 breakStmt().bind("break"),
                 expr(ignoringImplicit(cxxThrowExpr().bind("throw")))));
  Finder->addMatcher(
      compoundStmt(forEach(
          ifStmt(unless(isConstexpr()),
                 hasThen(stmt(
                     anyOf(ControlFlowInterruptorMatcher,
                           compoundStmt(has(ControlFlowInterruptorMatcher))))),
                 hasElse(stmt().bind("else")))
              .bind("if"))),
      this);
}

void ElseAfterReturnCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *If = Result.Nodes.getNodeAs<IfStmt>("if");
  SourceLocation ElseLoc = If->getElseLoc();
  std::string ControlFlowInterruptor;
  for (const auto *BindingName : {"return", "continue", "break", "throw"})
    if (Result.Nodes.getNodeAs<Stmt>(BindingName))
      ControlFlowInterruptor = BindingName;

  DiagnosticBuilder Diag = diag(ElseLoc, "do not use 'else' after '%0'")
                           << ControlFlowInterruptor;
  Diag << tooling::fixit::createRemoval(ElseLoc);

  // FIXME: Removing the braces isn't always safe. Do a more careful analysis.
  // FIXME: Change clang-format to correctly un-indent the code.
  if (const auto *CS = Result.Nodes.getNodeAs<CompoundStmt>("else"))
    Diag << tooling::fixit::createRemoval(CS->getLBracLoc())
         << tooling::fixit::createRemoval(CS->getRBracLoc());
}

} // namespace readability
} // namespace tidy
} // namespace clang
