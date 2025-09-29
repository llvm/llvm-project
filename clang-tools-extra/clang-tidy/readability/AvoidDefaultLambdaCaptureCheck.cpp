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

// Find the source range of the default capture token (= or &)
SourceRange getDefaultCaptureRange(const LambdaExpr *Lambda,
                                   const SourceManager &SM,
                                   const LangOptions &LangOpts) {
  SourceLocation DefaultLoc = Lambda->getCaptureDefaultLoc();
  if (DefaultLoc.isInvalid())
    return SourceRange();

  // The default capture is a single token
  SourceLocation EndLoc =
      Lexer::getLocForEndOfToken(DefaultLoc, 0, SM, LangOpts);
  return SourceRange(DefaultLoc, EndLoc);
}

// Find where to insert implicit captures
SourceLocation getImplicitCaptureInsertionLoc(const LambdaExpr *Lambda,
                                              const SourceManager &SM,
                                              const LangOptions &LangOpts) {
  // If there are explicit captures, insert after the last one
  if (Lambda->explicit_capture_begin() != Lambda->explicit_capture_end()) {
    // Find the location after the last explicit capture
    const auto *LastExplicit = Lambda->explicit_capture_end() - 1;
    SourceLocation LastLoc = LastExplicit->getLocation();
    if (LastLoc.isValid()) {
      return Lexer::getLocForEndOfToken(LastLoc, 0, SM, LangOpts);
    }
  }

  // If no explicit captures, insert after the default capture
  SourceLocation DefaultLoc = Lambda->getCaptureDefaultLoc();
  if (DefaultLoc.isValid()) {
    return Lexer::getLocForEndOfToken(DefaultLoc, 0, SM, LangOpts);
  }

  // Fallback: insert at the beginning of the capture list
  return Lambda->getIntroducerRange().getBegin().getLocWithOffset(1);
}

// Generate the text for an implicit capture
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

  // TODO: handle VLAs and other weird captures
  return std::nullopt;
}

// Check if we need a comma before inserting captures
bool needsCommaBefore(const LambdaExpr *Lambda, SourceLocation InsertLoc,
                      const SourceManager &SM, const LangOptions &LangOpts) {
  // If there are explicit captures, we need a comma
  return Lambda->explicit_capture_begin() != Lambda->explicit_capture_end();
}

} // namespace

void AvoidDefaultLambdaCaptureCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(lambdaExpr(hasDefaultCapture()).bind("lambda"), this);
}

void AvoidDefaultLambdaCaptureCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  assert(Lambda);

  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = Result.Context->getLangOpts();

  const SourceLocation DefaultCaptureLoc = Lambda->getCaptureDefaultLoc();
  if (DefaultCaptureLoc.isInvalid())
    return;

  auto Diag = diag(DefaultCaptureLoc,
                   "lambda default captures are discouraged; "
                   "prefer to capture specific variables explicitly");

  // Get the range of the default capture token to remove
  SourceRange DefaultRange = getDefaultCaptureRange(Lambda, SM, LangOpts);
  if (!DefaultRange.isValid())
    return;

  // Collect all implicit captures that need to be made explicit
  std::vector<std::string> ImplicitCaptureTexts;
  for (const auto &Capture : Lambda->implicit_captures()) {
    if (const auto CaptureText = generateImplicitCaptureText(Capture)) {
      ImplicitCaptureTexts.push_back(CaptureText.value());
    }
  }

  // If there are no implicit captures, just remove the default capture
  if (ImplicitCaptureTexts.empty()) {
    // Also remove any trailing comma if it exists
    SourceLocation AfterDefault = DefaultRange.getEnd();
    SourceLocation CommaLoc = Lexer::findLocationAfterToken(
        AfterDefault, tok::comma, SM, LangOpts, false);

    if (CommaLoc.isValid()) {
      // Remove default capture and the comma
      SourceRange RemovalRange(DefaultRange.getBegin(), CommaLoc);
      Diag << FixItHint::CreateRemoval(RemovalRange);
    } else {
      // Just remove the default capture
      Diag << FixItHint::CreateRemoval(DefaultRange);
    }
    return;
  }

  // Find where to insert the implicit captures
  SourceLocation InsertLoc =
      getImplicitCaptureInsertionLoc(Lambda, SM, LangOpts);
  if (!InsertLoc.isValid())
    return;

  // Apply the transformations:
  // 1. Remove the default capture
  Diag << FixItHint::CreateRemoval(DefaultRange);

  // 2. Insert the explicit captures if any
  if (!ImplicitCaptureTexts.empty()) {
    // Build the insertion text for implicit captures
    std::string InsertionText;
    bool NeedsComma = needsCommaBefore(Lambda, InsertLoc, SM, LangOpts);

    for (size_t I = 0; I < ImplicitCaptureTexts.size(); ++I) {
      if (NeedsComma || I > 0) {
        InsertionText += ", ";
      }
      InsertionText += ImplicitCaptureTexts[I];
    }

    Diag << FixItHint::CreateInsertion(InsertLoc, InsertionText);
  }
}

} // namespace clang::tidy::readability
