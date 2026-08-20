//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidDefaultLambdaCaptureCheck.h"
#include "clang/AST/DeclBase.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/Lambda.h"
#include "llvm/ADT/StringExtras.h"
#include <cassert>
#include <string>
#include <vector>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

AST_MATCHER(LambdaExpr, hasDefaultCapture) {
  return Node.getCaptureDefault() != LCD_None;
}

// `isInStdNamespace()` only matches declarations whose immediate enclosing
// namespace is `std`, so it misses niebloids like `std::ranges::all_of` that
// live in a namespace nested inside `std`.
AST_MATCHER(Decl, isInStdNamespaceOrNested) {
  for (const DeclContext *DC = Node.getDeclContext(); DC != nullptr;
       DC = DC->getParent()) {
    if (DC->isStdNamespace())
      return true;
  }
  return false;
}

} // namespace

static std::string generateCaptureText(const LambdaCapture &Capture) {
  if (Capture.capturesThis())
    return Capture.getCaptureKind() == LCK_StarThis ? "*this" : "this";

  std::string Result;
  if (Capture.getCaptureKind() == LCK_ByRef)
    Result += '&';

  Result += Capture.getCapturedVar()->getName().str();
  return Result;
}

AvoidDefaultLambdaCaptureCheck::AvoidDefaultLambdaCaptureCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreInSTL(Options.get("IgnoreInSTL", false)) {}

void AvoidDefaultLambdaCaptureCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreInSTL", IgnoreInSTL);
}

void AvoidDefaultLambdaCaptureCheck::registerMatchers(MatchFinder *Finder) {
  if (IgnoreInSTL) {
    auto StdFunctionCall =
        callExpr(callee(functionDecl(isInStdNamespaceOrNested())));
    auto StdNiebloidCall = cxxOperatorCallExpr(
        hasOverloadedOperatorName("()"),
        hasArgument(0, declRefExpr(to(varDecl(isInStdNamespaceOrNested())))));

    Finder->addMatcher(
        lambdaExpr(
            hasDefaultCapture(),
            unless(hasAncestor(stmt(anyOf(StdFunctionCall, StdNiebloidCall)))))
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

  const auto Diag = diag(DefaultCaptureLoc,
                         "lambda uses default capture mode; explicitly capture "
                         "variables instead");

  // For template-dependent lambdas, the list of captures hasn't been created
  // yet, so the list of implicit captures is empty.
  if (ImplicitCaptures.empty() && Lambda->getLambdaClass()->isDependentType())
    return;

  const std::string ReplacementText = llvm::join(ImplicitCaptures, ", ");

  // Don't suggest a fix-it if the default capture is within a macro expansion,
  // as the replacement may not be correct for all uses of the macro
  if (!DefaultCaptureLoc.isMacroID()) {
    Diag << FixItHint::CreateReplacement(Lambda->getCaptureDefaultLoc(),
                                         ReplacementText);
  }
}

} // namespace clang::tidy::readability
