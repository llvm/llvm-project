//===--- MisleadingCaptureDefaultByValueCheck.cpp - clang-tidy-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MisleadingCaptureDefaultByValueCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

MisleadingCaptureDefaultByValueCheck::MisleadingCaptureDefaultByValueCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void MisleadingCaptureDefaultByValueCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(lambdaExpr(hasAnyCapture(capturesThis())).bind("lambda"),
                     this);
}

static SourceLocation findDefaultCaptureEnd(const LambdaExpr *Lambda,
                                            ASTContext &Context) {
  for (const LambdaCapture &Capture : Lambda->explicit_captures()) {
    if (Capture.isExplicit()) {
      if (Capture.getCaptureKind() == LCK_ByRef) {
        const SourceManager &SourceMgr = Context.getSourceManager();
        SourceLocation AddressofLoc = utils::lexer::findPreviousTokenKind(
            Capture.getLocation(), SourceMgr, Context.getLangOpts(), tok::amp);
        return AddressofLoc;
      }
      return Capture.getLocation();
    }
  }
  return Lambda->getIntroducerRange().getEnd();
}

static std::string createReplacementText(const LambdaExpr *Lambda) {
  std::string Replacement;
  llvm::raw_string_ostream Stream(Replacement);

  auto AppendName = [&](llvm::StringRef Name) {
    if (!Replacement.empty()) {
      Stream << ", ";
    }
    if (Lambda->getCaptureDefault() == LCD_ByRef && Name != "this") {
      Stream << "&" << Name;
    } else {
      Stream << Name;
    }
  };

  for (const LambdaCapture &Capture : Lambda->implicit_captures()) {
    assert(Capture.isImplicit());
    if (Capture.capturesVariable() && Capture.isImplicit()) {
      AppendName(Capture.getCapturedVar()->getName());
    } else if (Capture.capturesThis()) {
      AppendName("this");
    }
  }
  if (!Replacement.empty() &&
      Lambda->explicit_capture_begin() != Lambda->explicit_capture_end()) {
    // Add back separator if we are adding explicit capture variables.
    Stream << ", ";
  }
  return Replacement;
}

void MisleadingCaptureDefaultByValueCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  if (!Lambda)
    return;

  if (Lambda->getCaptureDefault() == LCD_ByCopy) {
    bool IsThisImplicitlyCaptured = std::any_of(
        Lambda->implicit_capture_begin(), Lambda->implicit_capture_end(),
        [](const LambdaCapture &Capture) { return Capture.capturesThis(); });
    auto Diag = diag(Lambda->getCaptureDefaultLoc(),
                     "lambdas that %select{|implicitly }0capture 'this' "
                     "should not specify a by-value capture default")
                << IsThisImplicitlyCaptured;

    std::string ReplacementText = createReplacementText(Lambda);
    SourceLocation DefaultCaptureEnd =
        findDefaultCaptureEnd(Lambda, *Result.Context);
    Diag << FixItHint::CreateReplacement(
        CharSourceRange::getCharRange(Lambda->getCaptureDefaultLoc(),
                                      DefaultCaptureEnd),
        ReplacementText);
  }
}

} // namespace clang::tidy::cppcoreguidelines
