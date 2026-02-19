//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseVectorUtilsCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

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

void UseVectorUtilsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *OuterCall = Result.Nodes.getNodeAs<CallExpr>("outer_call");
  assert(OuterCall);

  const auto *InnerCall = Result.Nodes.getNodeAs<CallExpr>("inner_call");
  assert(InnerCall);

  const auto *OuterCallee =
      cast<DeclRefExpr>(OuterCall->getCallee()->IgnoreImplicit());

  const StringRef InnerFuncName =
      cast<NamedDecl>(InnerCall->getCalleeDecl())->getName();

  // Determine the replacement function name (unqualified).
  const llvm::SmallDenseMap<StringRef, StringRef, 2>
      InnerFuncNameToReplacementFuncName = {
          {"map_range", "map_to_vector"},
          {"make_filter_range", "filter_to_vector"},
      };
  const StringRef ReplacementFuncName =
      InnerFuncNameToReplacementFuncName.lookup(InnerFuncName);
  assert(!ReplacementFuncName.empty() && "Unhandled function?");

  auto Diag = diag(OuterCall->getBeginLoc(), "use '%0'") << ReplacementFuncName;

  // Replace the outer function name (preserving qualifier and template args),
  // and then remove the inner call's callee and opening paren and closing
  // paren. Example:
  // ```
  // llvm::to_vector<4>(llvm::map_range(X, F))
  //       ^replace~^   ^----remove-----^   ^
  //                                      remove
  // ```
  const SourceManager &SM = *Result.SourceManager;
  const std::optional<Token> InnerLParen =
      utils::lexer::findNextTokenSkippingComments(
          InnerCall->getCallee()->getEndLoc(), SM, getLangOpts());
  if (!InnerLParen || InnerLParen->isNot(tok::l_paren))
    return; // Unexpected token, possibly a macro?

  Diag << FixItHint::CreateReplacement(
              OuterCallee->getNameInfo().getSourceRange(), ReplacementFuncName)
       << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
              InnerCall->getBeginLoc(), InnerLParen->getEndLoc()))
       << FixItHint::CreateRemoval(InnerCall->getRParenLoc());

  // Add include for `SmallVectorExtras.h` if needed.
  if (auto IncludeFixit = Inserter.createIncludeInsertion(
          SM.getFileID(OuterCall->getBeginLoc()),
          "llvm/ADT/SmallVectorExtras.h"))
    Diag << *IncludeFixit;
}

} // namespace clang::tidy::llvm_check
