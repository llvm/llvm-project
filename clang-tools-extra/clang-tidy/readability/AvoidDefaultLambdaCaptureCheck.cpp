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

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static std::string generateCaptureText(const LambdaCapture &Capture) {
  if (Capture.capturesThis())
    return Capture.getCaptureKind() == LCK_StarThis ? "*this" : "this";

  std::string Result;
  if (Capture.getCaptureKind() == LCK_ByRef)
    Result += "&";

  Result += Capture.getCapturedVar()->getName().str();
  return Result;
}

void AvoidDefaultLambdaCaptureCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(lambdaExpr(hasDefaultCapture()).bind("lambda"), this);
}

void AvoidDefaultLambdaCaptureCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  assert(Lambda);

  const SourceLocation DefaultCaptureLoc = Lambda->getCaptureDefaultLoc();
  if (DefaultCaptureLoc.isInvalid())
    return;

  std::vector<std::string> ImplicitCaptures;
  for (const LambdaCapture &Capture : Lambda->implicit_captures()) {
    // It is impossible to explicitly capture a VLA in C++, since VLAs don't
    // exist in ISO C++ and so the syntax was never created to capture them.
    if (Capture.getCaptureKind() == LCK_VLAType)
      return;
    ImplicitCaptures.push_back(generateCaptureText(Capture));
  }

  auto Diag = diag(DefaultCaptureLoc,
                   "lambda uses default capture mode; explicitly capture "
                   "variables instead");

  // For template-dependent lambdas, the list of captures hasn't been created
  // yet, so the list of implicit captures is empty.
  if (ImplicitCaptures.empty() && Lambda->isGenericLambda())
    return;

  const std::string ReplacementText = llvm::join(ImplicitCaptures, ", ");

  Diag << FixItHint::CreateReplacement(Lambda->getCaptureDefaultLoc(),
                                       ReplacementText);
}

} // namespace clang::tidy::readability
