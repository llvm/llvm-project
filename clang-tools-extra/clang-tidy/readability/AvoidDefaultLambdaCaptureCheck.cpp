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

namespace {
AST_MATCHER(LambdaExpr, hasDefaultCapture) {
  return Node.getCaptureDefault() != LCD_None;
}

std::string getCaptureString(const LambdaCapture &Capture) {
  if (Capture.capturesThis()) {
    return Capture.getCaptureKind() == LCK_StarThis ? "*this" : "this";
  }

  if (Capture.capturesVariable()) {
    std::string Result;
    if (Capture.getCaptureKind() == LCK_ByRef) {
      Result += "&";
    }
    Result += Capture.getCapturedVar()->getName().str();
    return Result;
  }

  // Handle VLA captures - these are rare but possible
  return "/* VLA capture */";
}

std::string buildExplicitCaptureList(const LambdaExpr *Lambda) {
  std::vector<std::string> CaptureStrings;

  // Add explicit captures first (preserve their order and syntax)
  for (const auto &Capture : Lambda->explicit_captures()) {
    CaptureStrings.push_back(getCaptureString(Capture));
  }

  // Add implicit captures (convert to explicit syntax)
  for (const auto &Capture : Lambda->implicit_captures()) {
    CaptureStrings.push_back(getCaptureString(Capture));
  }

  return "[" + llvm::join(CaptureStrings, ", ") + "]";
}

SourceRange getCaptureListRange(const LambdaExpr *Lambda) {
  SourceRange IntroducerRange = Lambda->getIntroducerRange();
  return IntroducerRange;
}

} // namespace

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

  // Build the replacement capture list
  std::string NewCaptureList = buildExplicitCaptureList(Lambda);

  // Get the range of the entire capture list [...]
  SourceRange CaptureListRange = getCaptureListRange(Lambda);

  auto Diag = diag(DefaultCaptureLoc,
                   "lambda default captures are discouraged; "
                   "prefer to capture specific variables explicitly");

  // Only provide fixit if we can determine a valid replacement
  if (CaptureListRange.isValid() && !NewCaptureList.empty()) {
    Diag << FixItHint::CreateReplacement(CaptureListRange, NewCaptureList);
  }
}

} // namespace clang::tidy::readability
