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

std::optional<std::string>
generateImplicitCaptureText(const LambdaCapture &Capture) {
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

  if (Capture.capturesVLAType()) {
    // VLA captures are rare and complex - for now we skip them
    // A full implementation would need to handle the VLA type properly
    return std::nullopt;
  }

  return std::nullopt;
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

  auto Diag = diag(DefaultCaptureLoc,
                   "lambda default captures are discouraged; "
                   "prefer to capture specific variables explicitly");

  // Build the complete replacement capture list
  std::vector<std::string> AllCaptures;

  // Add explicit captures first (preserve their order)
  for (const auto &Capture : Lambda->explicit_captures()) {
    if (const auto CaptureText = generateImplicitCaptureText(Capture)) {
      AllCaptures.push_back(CaptureText.value());
    }
  }

  // Add implicit captures (convert to explicit)
  for (const auto &Capture : Lambda->implicit_captures()) {
    if (const auto CaptureText = generateImplicitCaptureText(Capture)) {
      AllCaptures.push_back(CaptureText.value());
    }
  }

  // Build the final capture list
  std::string ReplacementText;
  if (AllCaptures.empty()) {
    ReplacementText = "[]";
  } else {
    ReplacementText = "[" + llvm::join(AllCaptures, ", ") + "]";
  }

  // Replace the entire capture list with the explicit version
  SourceRange IntroducerRange = Lambda->getIntroducerRange();
  if (IntroducerRange.isValid()) {
    Diag << FixItHint::CreateReplacement(IntroducerRange, ReplacementText);
  }
}

} // namespace clang::tidy::readability
