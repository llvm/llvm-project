//===--- CloexecCheck.cpp - clang-tidy-------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CloexecCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::android {

namespace {
// Helper function to form the correct string mode for Type3.
// Build the replace text. If it's string constant, add <Mode> directly in the
// end of the string. Else, add <Mode>.
std::string buildFixMsgForStringFlag(const Expr *Arg, const SourceManager &SM,
                                     const LangOptions &LangOpts, char Mode) {
  if (Arg->getBeginLoc().isMacroID())
    return (Lexer::getSourceText(
                CharSourceRange::getTokenRange(Arg->getSourceRange()), SM,
                LangOpts) +
            " \"" + Twine(Mode) + "\"")
        .str();

  StringRef SR = cast<StringLiteral>(Arg->IgnoreParenCasts())->getString();
  return ("\"" + SR + Twine(Mode) + "\"").str();
}
} // namespace

const char *CloexecCheck::FuncDeclBindingStr = "funcDecl";

const char *CloexecCheck::FuncBindingStr ="func";

void CloexecCheck::registerMatchersImpl(
    MatchFinder *Finder, internal::Matcher<FunctionDecl> Function) {
  // We assume all the checked APIs are C functions.
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(isExternC(), Function).bind(FuncDeclBindingStr)))
          .bind(FuncBindingStr),
      this);
}

void CloexecCheck::insertMacroFlag(const MatchFinder::MatchResult &Result,
                                   StringRef MacroFlag, int ArgPos) {
  const auto *MatchedCall = Result.Nodes.getNodeAs<CallExpr>(FuncBindingStr);
  const auto *FlagArg = MatchedCall->getArg(ArgPos);
  const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>(FuncDeclBindingStr);
  SourceManager &SM = *Result.SourceManager;

  if (utils::exprHasBitFlagWithSpelling(FlagArg->IgnoreParenCasts(), SM,
                                        Result.Context->getLangOpts(),
                                        MacroFlag))
    return;

  SourceLocation EndLoc =
      Lexer::getLocForEndOfToken(SM.getFileLoc(FlagArg->getEndLoc()), 0, SM,
                                 Result.Context->getLangOpts());

  diag(EndLoc, "%0 should use %1 where possible")
      << FD << MacroFlag
      << FixItHint::CreateInsertion(EndLoc, (Twine(" | ") + MacroFlag).str());
}

void CloexecCheck::replaceFunc(const MatchFinder::MatchResult &Result,
                               StringRef WarningMsg, StringRef FixMsg) {
  const auto *MatchedCall = Result.Nodes.getNodeAs<CallExpr>(FuncBindingStr);
  diag(MatchedCall->getBeginLoc(), WarningMsg)
      << FixItHint::CreateReplacement(MatchedCall->getSourceRange(), FixMsg);
}

void CloexecCheck::insertStringFlag(
    const ast_matchers::MatchFinder::MatchResult &Result, const char Mode,
    const int ArgPos) {
  const auto *MatchedCall = Result.Nodes.getNodeAs<CallExpr>(FuncBindingStr);
  const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>(FuncDeclBindingStr);
  const auto *ModeArg = MatchedCall->getArg(ArgPos);

  // Check if the <Mode> may be in the mode string.
  const auto *ModeStr = dyn_cast<StringLiteral>(ModeArg->IgnoreParenCasts());
  if (!ModeStr || ModeStr->getString().contains(Mode))
    return;

  std::string ReplacementText = buildFixMsgForStringFlag(
      ModeArg, *Result.SourceManager, Result.Context->getLangOpts(), Mode);

  diag(ModeArg->getBeginLoc(), "use %0 mode '%1' to set O_CLOEXEC")
      << FD << std::string(1, Mode)
      << FixItHint::CreateReplacement(ModeArg->getSourceRange(),
                                      ReplacementText);
}

StringRef CloexecCheck::getSpellingArg(const MatchFinder::MatchResult &Result,
                                       int N) const {
  const auto *MatchedCall = Result.Nodes.getNodeAs<CallExpr>(FuncBindingStr);
  const SourceManager &SM = *Result.SourceManager;
  return Lexer::getSourceText(
      CharSourceRange::getTokenRange(MatchedCall->getArg(N)->getSourceRange()),
      SM, Result.Context->getLangOpts());
}

} // namespace clang::tidy::android
