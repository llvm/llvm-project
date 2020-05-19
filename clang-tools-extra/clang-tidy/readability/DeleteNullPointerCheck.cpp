//===--- DeleteNullPointerCheck.cpp - clang-tidy---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeleteNullPointerCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

void DeleteNullPointerCheck::registerMatchers(MatchFinder *Finder) {
  const auto DeleteExpr =
      cxxDeleteExpr(has(castExpr(has(declRefExpr(
                        to(decl(equalsBoundNode("deletedPointer"))))))))
          .bind("deleteExpr");

  const auto DeleteMemberExpr =
      cxxDeleteExpr(has(castExpr(has(memberExpr(hasDeclaration(
                        fieldDecl(equalsBoundNode("deletedMemberPointer"))))))))
          .bind("deleteMemberExpr");

  const auto PointerExpr = ignoringImpCasts(anyOf(
      declRefExpr(to(decl().bind("deletedPointer"))),
      memberExpr(hasDeclaration(fieldDecl().bind("deletedMemberPointer")))));

  const auto PointerCondition = castExpr(hasCastKind(CK_PointerToBoolean),
                                         hasSourceExpression(PointerExpr));
  const auto BinaryPointerCheckCondition = binaryOperator(
      hasOperands(castExpr(hasCastKind(CK_NullToPointer)), PointerExpr));

  Finder->addMatcher(
      ifStmt(hasCondition(anyOf(PointerCondition, BinaryPointerCheckCondition)),
             hasThen(anyOf(
                 DeleteExpr, DeleteMemberExpr,
                 compoundStmt(anyOf(has(DeleteExpr), has(DeleteMemberExpr)),
                              statementCountIs(1))
                     .bind("compound"))))
          .bind("ifWithDelete"),
      this);
}

void DeleteNullPointerCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *IfWithDelete = Result.Nodes.getNodeAs<IfStmt>("ifWithDelete");
  const auto *Compound = Result.Nodes.getNodeAs<CompoundStmt>("compound");

  auto Diag = diag(
      IfWithDelete->getBeginLoc(),
      "'if' statement is unnecessary; deleting null pointer has no effect");
  if (IfWithDelete->getElse())
    return;
  // FIXME: generate fixit for this case.

  Diag << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
      IfWithDelete->getBeginLoc(),
      utils::lexer::getPreviousToken(IfWithDelete->getThen()->getBeginLoc(),
                                     *Result.SourceManager,
                                     Result.Context->getLangOpts())
          .getLocation()));

  if (Compound) {
    Diag << FixItHint::CreateRemoval(
        CharSourceRange::getTokenRange(Compound->getLBracLoc()));
    Diag << FixItHint::CreateRemoval(
        CharSourceRange::getTokenRange(Compound->getRBracLoc()));
  }
}

} // namespace readability
} // namespace tidy
} // namespace clang
