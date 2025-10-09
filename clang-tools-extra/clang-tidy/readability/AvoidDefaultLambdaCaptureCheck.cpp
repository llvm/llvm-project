//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidDefaultLambdaCaptureCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/Lambda.h"
#include "clang/Lex/Lexer.h"

using namespace clang::tidy::readability;

static std::string generateCaptureText(const clang::LambdaCapture &Capture) {
  if (Capture.capturesThis())
    return Capture.getCaptureKind() == clang::LCK_StarThis ? "*this" : "this";

  std::string Result;
  if (Capture.getCaptureKind() == clang::LCK_ByRef) {
    Result += "&";
  }
  Result += Capture.getCapturedVar()->getName().str();
  return Result;
}

void AvoidDefaultLambdaCaptureCheck::registerMatchers(
    clang::ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(
      clang::ast_matchers::lambdaExpr(clang::ast_matchers::hasDefaultCapture())
          .bind("lambda"),
      this);
}

void AvoidDefaultLambdaCaptureCheck::check(
    const clang::ast_matchers::MatchFinder::MatchResult &Result) {
  const auto *Lambda = Result.Nodes.getNodeAs<clang::LambdaExpr>("lambda");
  assert(Lambda);

  const clang::SourceLocation DefaultCaptureLoc =
      Lambda->getCaptureDefaultLoc();
  if (DefaultCaptureLoc.isInvalid())
    return;

  std::vector<std::string> ImplicitCaptures;
  for (const auto &Capture : Lambda->implicit_captures()) {
    // It is impossible to explicitly capture a VLA in C++, since VLAs don't
    // exist in ISO C++ and so the syntax was never created to capture them.
    if (Capture.getCaptureKind() == LCK_VLAType)
      return;
    ImplicitCaptures.push_back(generateCaptureText(Capture));
  }

  auto Diag = diag(DefaultCaptureLoc,
                   "lambda default captures are discouraged; "
                   "prefer to capture specific variables explicitly");

  // For template-dependent lambdas, the list of captures hasn't been created
  // yet, so the list of implicit captures is empty.
  if (ImplicitCaptures.empty() && Lambda->isGenericLambda())
    return;

  const auto ReplacementText = [&ImplicitCaptures]() {
    return llvm::join(ImplicitCaptures, ", ");
  }();

  Diag << clang::FixItHint::CreateReplacement(Lambda->getCaptureDefaultLoc(),
                                              ReplacementText);
}
