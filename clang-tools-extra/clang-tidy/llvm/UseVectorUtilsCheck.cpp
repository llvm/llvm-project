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
      Inserter(utils::IncludeSorter::IS_LLVM, areDiagsSelfContained()) {}

void UseVectorUtilsCheck::registerPPCallbacks(const SourceManager &SM,
                                              Preprocessor *PP,
                                              Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void UseVectorUtilsCheck::registerMatchers(MatchFinder *Finder) {
  // Match `llvm::to_vector(llvm::map_range(X, F))` or
  // `llvm::to_vector(llvm::make_filter_range(X, Pred))`.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("::llvm::to_vector"))),
               hasArgument(0, callExpr(callee(functionDecl(hasAnyName(
                                           "::llvm::map_range",
                                           "::llvm::make_filter_range"))),
                                       argumentCountIs(2))
                                  .bind("inner_call")),
               argumentCountIs(1))
          .bind("outer_call"),
      this);
}

// Returns the original qualifier spelling (e.g., `llvm::` or `::llvm::`) for
// the diagnostic message.
static StringRef getQualifierSpelling(const DeclRefExpr *DeclRef,
                                      const MatchFinder::MatchResult &Result) {
  if (const auto QualifierLoc = DeclRef->getQualifierLoc()) {
    return Lexer::getSourceText(
        CharSourceRange::getTokenRange(QualifierLoc.getSourceRange()),
        *Result.SourceManager, Result.Context->getLangOpts());
  }
  return "";
}

void UseVectorUtilsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *OuterCall = Result.Nodes.getNodeAs<CallExpr>("outer_call");
  assert(OuterCall);

  const auto *InnerCall = Result.Nodes.getNodeAs<CallExpr>("inner_call");
  assert(InnerCall);

  const auto *OuterCallee =
      cast<DeclRefExpr>(OuterCall->getCallee()->IgnoreImplicit());
  const auto *InnerCallee =
      cast<DeclRefExpr>(InnerCall->getCallee()->IgnoreImplicit());

  const StringRef InnerFuncName =
      cast<NamedDecl>(InnerCallee->getDecl())->getName();

  // Determine the replacement function name (unqualified).
  const llvm::SmallDenseMap<StringRef, StringRef, 2>
      InnerFuncNameToReplacementFuncName = {
          {"map_range", "map_to_vector"},
          {"make_filter_range", "filter_to_vector"},
      };
  const StringRef ReplacementFuncName =
      InnerFuncNameToReplacementFuncName.lookup(InnerFuncName);
  assert(!ReplacementFuncName.empty() && "Unhandled function?");

  auto Diag = diag(OuterCall->getBeginLoc(), "use '%0%1'")
              << getQualifierSpelling(OuterCallee, Result)
              << ReplacementFuncName;

  // Replace only the unqualified function name, preserving qualifier and
  // template arguments.
  const auto InnerCallUntilFirstArg = CharSourceRange::getCharRange(
      InnerCall->getBeginLoc(), InnerCall->getArg(0)->getBeginLoc());
  Diag << FixItHint::CreateReplacement(
              OuterCallee->getNameInfo().getSourceRange(), ReplacementFuncName)
       << FixItHint::CreateRemoval(InnerCallUntilFirstArg)
       << FixItHint::CreateRemoval(InnerCall->getRParenLoc());

  // Add include for `SmallVectorExtras.h` if needed.
  const SourceManager &SM = *Result.SourceManager;
  if (auto IncludeFixit = Inserter.createIncludeInsertion(
          SM.getFileID(OuterCall->getBeginLoc()),
          "llvm/ADT/SmallVectorExtras.h"))
    Diag << *IncludeFixit;
}

} // namespace clang::tidy::llvm_check
