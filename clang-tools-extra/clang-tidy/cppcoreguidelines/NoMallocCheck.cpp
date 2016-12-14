//===--- NoMallocCheck.cpp - clang-tidy------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NoMallocCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <iostream>
#include <string>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

void NoMallocCheck::registerMatchers(MatchFinder *Finder) {
  // C-style memory management is only problematic in C++.
  if (!getLangOpts().CPlusPlus)
    return;

  // Registering malloc, will suggest RAII.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasAnyName("::malloc", "::calloc"))))
          .bind("aquisition"),
      this);

  // Registering realloc calls, suggest std::vector or std::string.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("::realloc")))).bind("realloc"),
      this);

  // Registering free calls, will suggest RAII instead.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("::free")))).bind("free"), this);
}

void NoMallocCheck::check(const MatchFinder::MatchResult &Result) {
  const CallExpr *Call = nullptr;
  StringRef Recommendation;

  if ((Call = Result.Nodes.getNodeAs<CallExpr>("aquisition")))
    Recommendation = "consider a container or a smart pointer";
  else if ((Call = Result.Nodes.getNodeAs<CallExpr>("realloc")))
    Recommendation = "consider std::vector or std::string";
  else if ((Call = Result.Nodes.getNodeAs<CallExpr>("free")))
    Recommendation = "use RAII";

  assert(Call && "Unhandled binding in the Matcher");

  diag(Call->getLocStart(), "do not manage memory manually; %0")
      << Recommendation << SourceRange(Call->getLocStart(), Call->getLocEnd());
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
