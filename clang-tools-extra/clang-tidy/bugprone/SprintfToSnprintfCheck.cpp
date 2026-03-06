//===--- SprintfToSnprintfCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SprintfToSnprintfCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

SprintfToSnprintfCheck::SprintfToSnprintfCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SprintfLikeFunctions(utils::options::parseStringList(
          Options.get("SprintfLikeFunctions", "::sprintf;::std::sprintf"))) {}

void SprintfToSnprintfCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SprintfLikeFunctions",
                utils::options::serializeStringList(SprintfLikeFunctions));
}

void SprintfToSnprintfCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasAnyName(SprintfLikeFunctions))),
               hasArgument(
                   0, ignoringParenImpCasts(
                          declRefExpr(to(varDecl(hasType(constantArrayType()))
                                             .bind("buffer")))
                              .bind("arg0"))))
          .bind("call"),
      this);
}

void SprintfToSnprintfCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  const auto *Buffer = Result.Nodes.getNodeAs<VarDecl>("buffer");

  if (!Call || !Buffer)
    return;

  const FunctionDecl *Callee = Call->getDirectCallee();
  if (!Callee)
    return;

  const StringRef FuncName = Callee->getName();
  const StringRef BufferName = Buffer->getName();

  auto Diag =
      diag(Call->getBeginLoc(), "use 'snprintf' instead of '%0' for fixed-size "
                                "character arrays")
      << FuncName;

  // Only provide an automated Fix-It if the function is exactly "sprintf"
  if (FuncName == "sprintf") {
    const SourceLocation FuncNameLoc = Call->getExprLoc();
    Diag << FixItHint::CreateReplacement(FuncNameLoc, "snprintf");

    const SourceLocation InsertLoc = Call->getArg(1)->getBeginLoc();
    const std::string SizeArg = "sizeof(" + BufferName.str() + "), ";
    Diag << FixItHint::CreateInsertion(InsertLoc, SizeArg);
  }
}

} // namespace clang::tidy::bugprone
