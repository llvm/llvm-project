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

/// Matches lambda expressions that have default capture modes.
///
/// Given
/// \code
///   auto l1 = [=]() {};  // matches
///   auto l2 = [&]() {};  // matches
///   auto l3 = []() {};   // does not match
/// \endcode
/// lambdaExpr(hasDefaultCapture())
///   matches l1 and l2, but not l3.
AST_MATCHER(LambdaExpr, hasDefaultCapture) {
  return Node.getCaptureDefault() != LCD_None;
}

} // namespace

static std::string generateCaptureText(const LambdaCapture &Capture) {
  if (Capture.capturesThis())
    return Capture.getCaptureKind() == LCK_StarThis ? "*this" : "this";

  std::string Result;
  if (Capture.getCaptureKind() == LCK_ByRef)
    Result += "&";

  Result += Capture.getCapturedVar()->getName().str();
  return Result;
}

AvoidDefaultLambdaCaptureCheck::AvoidDefaultLambdaCaptureCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreImplicitCapturesInSTL(
          Options.get("IgnoreImplicitCapturesInSTL", false)) {}

void AvoidDefaultLambdaCaptureCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreImplicitCapturesInSTL",
                IgnoreImplicitCapturesInSTL);
}

void AvoidDefaultLambdaCaptureCheck::registerMatchers(MatchFinder *Finder) {
  if (IgnoreImplicitCapturesInSTL) {
    Finder->addMatcher(lambdaExpr(hasDefaultCapture(),
                                  unless(hasAncestor(callExpr(callee(
                                      functionDecl(isInStdNamespace()))))))
                           .bind("lambda"),
                       this);
  } else {
    Finder->addMatcher(lambdaExpr(hasDefaultCapture()).bind("lambda"), this);
  }
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
  if (ImplicitCaptures.empty() && Lambda->getLambdaClass()->isDependentType())
    return;

  const std::string ReplacementText = llvm::join(ImplicitCaptures, ", ");

  // Don't suggest a fix-it if the default capture is within a macro expansion,
  // as the replacement may not be correct for all uses of the macro
  const SourceManager &SM = *Result.SourceManager;
  if (!SM.isMacroBodyExpansion(DefaultCaptureLoc) &&
      !SM.isMacroArgExpansion(DefaultCaptureLoc)) {
    Diag << FixItHint::CreateReplacement(Lambda->getCaptureDefaultLoc(),
                                         ReplacementText);
  }
}

} // namespace clang::tidy::readability
