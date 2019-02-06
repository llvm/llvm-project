//===--- ReplaceRandomShuffleCheck.cpp - clang-tidy------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReplaceRandomShuffleCheck.h"
#include "../utils/FixItHintUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

ReplaceRandomShuffleCheck::ReplaceRandomShuffleCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeStyle(utils::IncludeSorter::parseIncludeStyle(
          Options.getLocalOrGlobal("IncludeStyle", "llvm"))) {}

void ReplaceRandomShuffleCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus11)
    return;

  const auto Begin = hasArgument(0, expr());
  const auto End = hasArgument(1, expr());
  const auto RandomFunc = hasArgument(2, expr().bind("randomFunc"));
  Finder->addMatcher(
      callExpr(anyOf(allOf(Begin, End, argumentCountIs(2)),
                     allOf(Begin, End, RandomFunc, argumentCountIs(3))),
               hasDeclaration(functionDecl(hasName("::std::random_shuffle"))),
               has(implicitCastExpr(has(declRefExpr().bind("name")))))
          .bind("match"),
      this);
}

void ReplaceRandomShuffleCheck::registerPPCallbacks(
    CompilerInstance &Compiler) {
  IncludeInserter = llvm::make_unique<utils::IncludeInserter>(
      Compiler.getSourceManager(), Compiler.getLangOpts(), IncludeStyle);
  Compiler.getPreprocessor().addPPCallbacks(
      IncludeInserter->CreatePPCallbacks());
}

void ReplaceRandomShuffleCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle",
                utils::IncludeSorter::toString(IncludeStyle));
}

void ReplaceRandomShuffleCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<DeclRefExpr>("name");
  const auto *MatchedArgumentThree = Result.Nodes.getNodeAs<Expr>("randomFunc");
  const auto *MatchedCallExpr = Result.Nodes.getNodeAs<CallExpr>("match");

  if (MatchedCallExpr->getBeginLoc().isMacroID())
    return;

  auto Diag = [&] {
    if (MatchedCallExpr->getNumArgs() == 3) {
      auto DiagL =
          diag(MatchedCallExpr->getBeginLoc(),
               "'std::random_shuffle' has been removed in C++17; use "
               "'std::shuffle' and an alternative random mechanism instead");
      DiagL << FixItHint::CreateReplacement(
          MatchedArgumentThree->getSourceRange(),
          "std::mt19937(std::random_device()())");
      return DiagL;
    } else {
      auto DiagL = diag(MatchedCallExpr->getBeginLoc(),
                        "'std::random_shuffle' has been removed in C++17; use "
                        "'std::shuffle' instead");
      DiagL << FixItHint::CreateInsertion(
          MatchedCallExpr->getRParenLoc(),
          ", std::mt19937(std::random_device()())");
      return DiagL;
    }
  }();

  std::string NewName = "shuffle";
  StringRef ContainerText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(MatchedDecl->getSourceRange()),
      *Result.SourceManager, getLangOpts());
  if (ContainerText.startswith("std::"))
    NewName = "std::" + NewName;

  Diag << FixItHint::CreateRemoval(MatchedDecl->getSourceRange());
  Diag << FixItHint::CreateInsertion(MatchedDecl->getBeginLoc(), NewName);

  if (Optional<FixItHint> IncludeFixit =
          IncludeInserter->CreateIncludeInsertion(
              Result.Context->getSourceManager().getFileID(
                  MatchedCallExpr->getBeginLoc()),
              "random", /*IsAngled=*/true))
    Diag << IncludeFixit.getValue();
}

} // namespace modernize
} // namespace tidy
} // namespace clang
