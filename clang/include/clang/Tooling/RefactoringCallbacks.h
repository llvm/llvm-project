//===--- RefactoringCallbacks.h - Structural query framework ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Provides callbacks to make common kinds of refactorings easy.
//
//  The general idea is to construct a matcher expression that describes a
//  subtree match on the AST and then replace the corresponding source code
//  either by some specific text or some other AST node.
//
//  Example:
//  int main(int argc, char **argv) {
//    ClangTool Tool(argc, argv);
//    MatchFinder Finder;
//    ReplaceStmtWithText Callback("integer", "42");
//    Finder.AddMatcher(id("integer", expression(integerLiteral())), Callback);
//    return Tool.run(newFrontendActionFactory(&Finder));
//  }
//
//  This will replace all integer literals with "42".
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTORING_CALLBACKS_H
#define LLVM_CLANG_TOOLING_REFACTORING_CALLBACKS_H

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

namespace clang {
namespace tooling {

/// \brief Base class for RefactoringCallbacks.
///
/// Collects \c tooling::Replacements while running.
class RefactoringCallback : public ast_matchers::MatchFinder::MatchCallback {
public:
  RefactoringCallback();
  Replacements &getReplacements();

protected:
  Replacements Replace;
};

/// \brief Replace the text of the statement bound to \c FromId with the text in
/// \c ToText.
class ReplaceStmtWithText : public RefactoringCallback {
public:
  ReplaceStmtWithText(StringRef FromId, StringRef ToText);
  virtual void run(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  std::string FromId;
  std::string ToText;
};

/// \brief Replace the text of the statement bound to \c FromId with the text of
/// the statement bound to \c ToId.
class ReplaceStmtWithStmt : public RefactoringCallback {
public:
  ReplaceStmtWithStmt(StringRef FromId, StringRef ToId);
  virtual void run(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  std::string FromId;
  std::string ToId;
};

/// \brief Replace an if-statement bound to \c Id with the outdented text of its
/// body, choosing the consequent or the alternative based on whether
/// \c PickTrueBranch is true.
class ReplaceIfStmtWithItsBody : public RefactoringCallback {
public:
  ReplaceIfStmtWithItsBody(StringRef Id, bool PickTrueBranch);
  virtual void run(const ast_matchers::MatchFinder::MatchResult &Result);

private:
  std::string Id;
  const bool PickTrueBranch;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTORING_CALLBACKS_H
