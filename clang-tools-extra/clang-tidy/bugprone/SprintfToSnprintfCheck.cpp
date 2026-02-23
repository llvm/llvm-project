//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SprintfToSnprintfCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void SprintfToSnprintfCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(hasName("::sprintf"))),
          hasArgument(
              0, ignoringParenImpCasts(declRefExpr(to(
                     varDecl(hasType(constantArrayType())).bind("buffer")
                 )).bind("arg0"))
          )
      ).bind("call"),
      this);
}

void SprintfToSnprintfCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  const auto *Buffer = Result.Nodes.getNodeAs<VarDecl>("buffer");

  if (!Call || !Buffer)
    return;

  StringRef BufferName = Buffer->getName();

  auto Diag = diag(Call->getBeginLoc(), "use 'snprintf' instead of 'sprintf' for fixed-size character arrays");

  SourceLocation FuncNameLoc = Call->getExprLoc();
  Diag << FixItHint::CreateReplacement(FuncNameLoc, "snprintf");

  SourceLocation InsertLoc = Call->getArg(1)->getBeginLoc();
  std::string SizeArg = "sizeof(" + BufferName.str() + "), ";
  Diag << FixItHint::CreateInsertion(InsertLoc, SizeArg);
}

} // namespace clang::tidy::bugprone
