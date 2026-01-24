//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseVectorUtilsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

UseVectorUtilsCheck::UseVectorUtilsCheck(StringRef Name,
                                         ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()) {}

void UseVectorUtilsCheck::registerPPCallbacks(const SourceManager &SM,
                                              Preprocessor *PP,
                                              Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void UseVectorUtilsCheck::registerMatchers(MatchFinder *Finder) {
  // Match `llvm::to_vector(llvm::map_range(X, F))`.
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(hasName("::llvm::to_vector"))),
          hasArgument(
              0, callExpr(callee(functionDecl(hasName("::llvm::map_range"))))
                     .bind("inner_call")))
          .bind("map_range_call"),
      this);

  // Match `llvm::to_vector(llvm::make_filter_range(X, Pred))`.
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(hasName("::llvm::to_vector"))),
          hasArgument(0, callExpr(callee(functionDecl(
                                      hasName("::llvm::make_filter_range"))))
                             .bind("inner_call")))
          .bind("filter_range_call"),
      this);
}

void UseVectorUtilsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MapRangeCall =
      Result.Nodes.getNodeAs<CallExpr>("map_range_call");
  const auto *FilterRangeCall =
      Result.Nodes.getNodeAs<CallExpr>("filter_range_call");
  if (!MapRangeCall && !FilterRangeCall)
    return;

  const auto *InnerCall = Result.Nodes.getNodeAs<CallExpr>("inner_call");
  assert(InnerCall && "inner_call must be bound if map_range_call or "
                      "filter_range_call matched");
  // Only handle the 2-argument overloads of `map_range`/`make_filter_range`, to
  // future-proof against additional overloads.
  if (InnerCall->getNumArgs() != 2)
    return;

  const CallExpr *OuterCall = MapRangeCall ? MapRangeCall : FilterRangeCall;

  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = getLangOpts();

  // Determine the base replacement function name.
  const StringRef ReplacementFuncBase =
      MapRangeCall ? "llvm::map_to_vector" : "llvm::filter_to_vector";
  const StringRef InnerFuncName =
      MapRangeCall ? "llvm::map_range" : "llvm::make_filter_range";

  // Check if `to_vector` was called with an explicit size template argument.
  std::string SizeTemplateArg;
  if (const auto *DRE =
          dyn_cast<DeclRefExpr>(OuterCall->getCallee()->IgnoreImplicit())) {
    if (DRE->hasExplicitTemplateArgs()) {
      // Extract the template argument text (e.g., `<4>`).
      const auto TemplateArgsCharRange = CharSourceRange::getTokenRange(
          DRE->getLAngleLoc(), DRE->getRAngleLoc());
      SizeTemplateArg =
          Lexer::getSourceText(TemplateArgsCharRange, SM, LangOpts).str();
    }
  }

  const std::string ReplacementFunc =
      (ReplacementFuncBase + SizeTemplateArg).str();
  const std::string ToVectorFunc = "llvm::to_vector" + SizeTemplateArg;

  // Build the replacement: Replace the whole expression with the new function
  // and the arguments from the inner call.
  auto Diag = diag(OuterCall->getBeginLoc(),
                   "use '%0' instead of '%1(%2(...))'")
              << ReplacementFunc << ToVectorFunc << InnerFuncName;

  // Get the range argument.
  const SourceRange RangeArgRange = InnerCall->getArg(0)->getSourceRange();
  const auto RangeArgCharRange = CharSourceRange::getTokenRange(RangeArgRange);
  const StringRef RangeArgText =
      Lexer::getSourceText(RangeArgCharRange, SM, LangOpts);

  // Get the function/predicate argument.
  const SourceRange FuncArgRange = InnerCall->getArg(1)->getSourceRange();
  const auto FuncArgCharRange = CharSourceRange::getTokenRange(FuncArgRange);
  const StringRef FuncArgText =
      Lexer::getSourceText(FuncArgCharRange, SM, LangOpts);

  // Create the replacement text.
  const std::string Replacement =
      (ReplacementFunc + "(" + RangeArgText + ", " + FuncArgText + ")").str();

  Diag << FixItHint::CreateReplacement(OuterCall->getSourceRange(), Replacement);

  // Add include for `SmallVectorExtras.h` if needed.
  if (auto IncludeFixit = Inserter.createIncludeInsertion(
          SM.getFileID(OuterCall->getBeginLoc()),
          "llvm/ADT/SmallVectorExtras.h"))
    Diag << *IncludeFixit;
}

} // namespace clang::tidy::llvm_check
