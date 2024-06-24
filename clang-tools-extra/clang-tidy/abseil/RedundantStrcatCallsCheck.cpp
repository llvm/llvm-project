//===--- RedundantStrcatCallsCheck.cpp - clang-tidy------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantStrcatCallsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <deque>

using namespace clang::ast_matchers;

namespace clang::tidy::abseil {

// TODO: Features to add to the check:
//  - Make it work if num_args > 26.
//  - Remove empty literal string arguments.
//  - Collapse consecutive literal string arguments into one (remove the ,).
//  - Replace StrCat(a + b)  ->  StrCat(a, b)  if a or b are strings.
//  - Make it work in macros if the outer and inner StrCats are both in the
//    argument.

void RedundantStrcatCallsCheck::registerMatchers(MatchFinder* Finder) {
  const auto CallToStrcat =
      callExpr(callee(functionDecl(hasName("::absl::StrCat"))));
  const auto CallToStrappend =
      callExpr(callee(functionDecl(hasName("::absl::StrAppend"))));
  // Do not match StrCat() calls that are descendants of other StrCat calls.
  // Those are handled on the ancestor call.
  const auto CallToEither = callExpr(
      callee(functionDecl(hasAnyName("::absl::StrCat", "::absl::StrAppend"))));
  Finder->addMatcher(
      callExpr(CallToStrcat, unless(hasAncestor(CallToEither))).bind("StrCat"),
      this);
  Finder->addMatcher(CallToStrappend.bind("StrAppend"), this);
}

namespace {

struct StrCatCheckResult {
  int NumCalls = 0;
  std::vector<FixItHint> Hints;
};

void removeCallLeaveArgs(const CallExpr *Call, StrCatCheckResult *CheckResult) {
  if (Call->getNumArgs() == 0)
    return;
  // Remove 'Foo('
  CheckResult->Hints.push_back(
      FixItHint::CreateRemoval(CharSourceRange::getCharRange(
          Call->getBeginLoc(), Call->getArg(0)->getBeginLoc())));
  // Remove the ')'
  CheckResult->Hints.push_back(
      FixItHint::CreateRemoval(CharSourceRange::getCharRange(
          Call->getRParenLoc(), Call->getEndLoc().getLocWithOffset(1))));
}

const clang::CallExpr *processArgument(const Expr *Arg,
                                       const MatchFinder::MatchResult &Result,
                                       StrCatCheckResult *CheckResult) {
  const auto IsAlphanum = hasDeclaration(cxxMethodDecl(hasName("AlphaNum")));
  static const auto* const Strcat = new auto(hasName("::absl::StrCat"));
  const auto IsStrcat = cxxBindTemporaryExpr(
      has(callExpr(callee(functionDecl(*Strcat))).bind("StrCat")));
  if (const auto *SubStrcatCall = selectFirst<const CallExpr>(
          "StrCat",
          match(stmt(traverse(TK_AsIs,
                              anyOf(cxxConstructExpr(IsAlphanum,
                                                     hasArgument(0, IsStrcat)),
                                    IsStrcat))),
                *Arg->IgnoreParenImpCasts(), *Result.Context))) {
    removeCallLeaveArgs(SubStrcatCall, CheckResult);
    return SubStrcatCall;
  }
  return nullptr;
}

StrCatCheckResult processCall(const CallExpr *RootCall, bool IsAppend,
                              const MatchFinder::MatchResult &Result) {
  StrCatCheckResult CheckResult;
  std::deque<const CallExpr*> CallsToProcess = {RootCall};

  while (!CallsToProcess.empty()) {
    ++CheckResult.NumCalls;

    const CallExpr* CallExpr = CallsToProcess.front();
    CallsToProcess.pop_front();

    int StartArg = CallExpr == RootCall && IsAppend;
    for (const auto *Arg : CallExpr->arguments()) {
      if (StartArg-- > 0) 
      	continue;
      if (const clang::CallExpr *Sub =
              processArgument(Arg, Result, &CheckResult)) {
        CallsToProcess.push_back(Sub);
      }
    }
  }
  return CheckResult;
}
}  // namespace

void RedundantStrcatCallsCheck::check(const MatchFinder::MatchResult& Result) {
  bool IsAppend = false;

  const CallExpr *RootCall = nullptr;
  if ((RootCall = Result.Nodes.getNodeAs<CallExpr>("StrCat"))) 
  	IsAppend = false;
  else if ((RootCall = Result.Nodes.getNodeAs<CallExpr>("StrAppend"))) 
  	IsAppend = true;
  else 
  	return;

  if (RootCall->getBeginLoc().isMacroID()) {
    // Ignore calls within macros.
    // In many cases the outer StrCat part of the macro and the inner StrCat is
    // a macro argument. Removing the inner StrCat() converts one macro
    // argument into many.
    return;
  }

  const StrCatCheckResult CheckResult = processCall(RootCall, IsAppend, Result);
  if (CheckResult.NumCalls == 1) {
    // Just one call, so nothing to fix.
    return;
  }

  diag(RootCall->getBeginLoc(), 
  	   "multiple calls to 'absl::StrCat' can be flattened into a single call")
      << CheckResult.Hints;
}

} // namespace clang::tidy::abseil
