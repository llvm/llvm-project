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

static std::optional<std::string>
generateImplicitCaptureText(const clang::LambdaCapture &Capture) {
  if (Capture.capturesThis()) {
    return Capture.getCaptureKind() == clang::LCK_StarThis ? "*this" : "this";
  }

  if (Capture.capturesVariable()) {
    std::string Result;
    if (Capture.getCaptureKind() == clang::LCK_ByRef) {
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

  auto Diag = diag(DefaultCaptureLoc,
                   "lambda default captures are discouraged; "
                   "prefer to capture specific variables explicitly");

  std::vector<std::string> AllCaptures;

  for (const auto &Capture : Lambda->explicit_captures()) {
    if (const auto CaptureText = generateImplicitCaptureText(Capture)) {
      AllCaptures.push_back(CaptureText.value());
    }
  }

  for (const auto &Capture : Lambda->implicit_captures()) {
    if (const auto CaptureText = generateImplicitCaptureText(Capture)) {
      AllCaptures.push_back(CaptureText.value());
    }
  }

  // Replace with new capture list
  std::string ReplacementText;
  if (AllCaptures.empty()) {
    ReplacementText = "[]";
  } else {
    ReplacementText = "[" + llvm::join(AllCaptures, ", ") + "]";
  }

  clang::SourceRange IntroducerRange = Lambda->getIntroducerRange();
  if (IntroducerRange.isValid()) {
    Diag << clang::FixItHint::CreateReplacement(IntroducerRange,
                                                ReplacementText);
  }
}
