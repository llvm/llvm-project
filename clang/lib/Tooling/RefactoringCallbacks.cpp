//===--- RefactoringCallbacks.cpp - Structural query framework ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/RefactoringCallbacks.h"

namespace clang {
namespace tooling {

RefactoringCallback::RefactoringCallback() {}
tooling::Replacements &RefactoringCallback::getReplacements() {
  return Replace;
}

static Replacement replaceStmtWithText(SourceManager &Sources,
                                       const Stmt &From,
                                       StringRef Text) {
  return tooling::Replacement(Sources, CharSourceRange::getTokenRange(
      From.getSourceRange()), Text);
}
static Replacement replaceStmtWithStmt(SourceManager &Sources,
                                       const Stmt &From,
                                       const Stmt &To) {
  return replaceStmtWithText(Sources, From, Lexer::getSourceText(
      CharSourceRange::getTokenRange(To.getSourceRange()),
      Sources, LangOptions()));
}

ReplaceStmtWithText::ReplaceStmtWithText(StringRef FromId, StringRef ToText)
    : FromId(FromId), ToText(ToText) {}

void ReplaceStmtWithText::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const Stmt *FromMatch = Result.Nodes.getStmtAs<Stmt>(FromId)) {
    Replace.insert(tooling::Replacement(
        *Result.SourceManager,
        CharSourceRange::getTokenRange(FromMatch->getSourceRange()),
        ToText));
  }
}

ReplaceStmtWithStmt::ReplaceStmtWithStmt(StringRef FromId, StringRef ToId)
    : FromId(FromId), ToId(ToId) {}

void ReplaceStmtWithStmt::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const Stmt *FromMatch = Result.Nodes.getStmtAs<Stmt>(FromId);
  const Stmt *ToMatch = Result.Nodes.getStmtAs<Stmt>(ToId);
  if (FromMatch && ToMatch)
    Replace.insert(replaceStmtWithStmt(
        *Result.SourceManager, *FromMatch, *ToMatch));
}

ReplaceIfStmtWithItsBody::ReplaceIfStmtWithItsBody(StringRef Id,
                                                   bool PickTrueBranch)
    : Id(Id), PickTrueBranch(PickTrueBranch) {}

void ReplaceIfStmtWithItsBody::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const IfStmt *Node = Result.Nodes.getStmtAs<IfStmt>(Id)) {
    const Stmt *Body = PickTrueBranch ? Node->getThen() : Node->getElse();
    if (Body) {
      Replace.insert(replaceStmtWithStmt(*Result.SourceManager, *Node, *Body));
    } else if (!PickTrueBranch) {
      // If we want to use the 'else'-branch, but it doesn't exist, delete
      // the whole 'if'.
      Replace.insert(replaceStmtWithText(*Result.SourceManager, *Node, ""));
    }
  }
}

} // end namespace tooling
} // end namespace clang
