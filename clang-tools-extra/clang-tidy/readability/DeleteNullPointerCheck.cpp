//===--- DeleteNullPointerCheck.cpp - clang-tidy---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DeleteNullPointerCheck.h"
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
  const auto BinaryPointerCheckCondition =
      binaryOperator(hasEitherOperand(castExpr(hasCastKind(CK_NullToPointer))),
                     hasEitherOperand(PointerExpr));

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
      IfWithDelete->getLocStart(),
      "'if' statement is unnecessary; deleting null pointer has no effect");
  if (IfWithDelete->getElse())
    return;
  // FIXME: generate fixit for this case.

  Diag << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
      IfWithDelete->getLocStart(),
      Lexer::getLocForEndOfToken(IfWithDelete->getCond()->getLocEnd(), 0,
                                 *Result.SourceManager,
                                 Result.Context->getLangOpts())));
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
