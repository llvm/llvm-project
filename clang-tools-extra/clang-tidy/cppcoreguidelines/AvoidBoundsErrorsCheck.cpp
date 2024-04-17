//===--- AvoidBoundsErrorsCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidBoundsErrorsCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

#include <iostream>
using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

bool isApplicable(const QualType &Type) {
  const auto TypeStr = Type.getAsString();
  bool Result = false;
  // Only check for containers in the std namespace
  if (TypeStr.find("std::vector") != std::string::npos) {
    Result = true;
  }
  if (TypeStr.find("std::array") != std::string::npos) {
    Result = true;
  }
  if (TypeStr.find("std::deque") != std::string::npos) {
    Result = true;
  }
  if (TypeStr.find("std::map") != std::string::npos) {
    Result = true;
  }
  if (TypeStr.find("std::unordered_map") != std::string::npos) {
    Result = true;
  }
  if (TypeStr.find("std::flat_map") != std::string::npos) {
    Result = true;
  }
  // TODO Add std::span with C++26
  return Result;
}

void AvoidBoundsErrorsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(callee(cxxMethodDecl(hasName("operator[]")).bind("f")))
          .bind("x"),
      this);
}

void AvoidBoundsErrorsCheck::check(const MatchFinder::MatchResult &Result) {
  const ASTContext &Context = *Result.Context;
  const SourceManager &Source = Context.getSourceManager();
  const auto *MatchedExpr = Result.Nodes.getNodeAs<CallExpr>("x");
  const auto *MatchedFunction = Result.Nodes.getNodeAs<CXXMethodDecl>("f");
  const auto Type = MatchedFunction->getThisType();
  if (!isApplicable(Type)) {
    return;
  }

  // Get original code.
  const SourceLocation b(MatchedExpr->getBeginLoc());
  const SourceLocation e(MatchedExpr->getEndLoc());
  const std::string OriginalCode =
      Lexer::getSourceText(CharSourceRange::getTokenRange(b, e), Source,
                           getLangOpts())
          .str();
  const auto Range = SourceRange(b, e);

  // Build replacement.
  std::string NewCode = OriginalCode;
  const auto BeginOpen = NewCode.find("[");
  NewCode.replace(BeginOpen, 1, ".at(");
  const auto BeginClose = NewCode.find("]");
  NewCode.replace(BeginClose, 1, ")");

  diag(MatchedExpr->getBeginLoc(), "Do not use operator[], use at() instead.")
      << FixItHint::CreateReplacement(Range, NewCode);
}

} // namespace clang::tidy::cppcoreguidelines
