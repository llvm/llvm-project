//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidReturnWithVoidValueCheck.h"
#include "../utils/BracesAroundStatement.h"
#include "../utils/LexerUtils.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static constexpr char IgnoreMacrosName[] = "IgnoreMacros";
static const bool IgnoreMacrosDefault = true;

static constexpr char StrictModeName[] = "StrictMode";
static const bool StrictModeDefault = true;

AvoidReturnWithVoidValueCheck::AvoidReturnWithVoidValueCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreMacros(Options.get(IgnoreMacrosName, IgnoreMacrosDefault)),
      StrictMode(Options.get(StrictModeName, StrictModeDefault)) {}

void AvoidReturnWithVoidValueCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      returnStmt(
          hasReturnValue(allOf(hasType(voidType()), unless(initListExpr()))),
          optionally(hasParent(
              compoundStmt(
                  optionally(hasParent(functionDecl().bind("function_parent"))))
                  .bind("compound_parent"))))
          .bind("void_return"),
      this);
}

void AvoidReturnWithVoidValueCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *VoidReturn = Result.Nodes.getNodeAs<ReturnStmt>("void_return");
  if (IgnoreMacros && VoidReturn->getBeginLoc().isMacroID())
    return;
  const auto *SurroundingBlock =
      Result.Nodes.getNodeAs<CompoundStmt>("compound_parent");
  if (!StrictMode && !SurroundingBlock)
    return;
  DiagnosticBuilder Diag = diag(VoidReturn->getBeginLoc(),
                                "return statement within a void function "
                                "should not have a specified return value");
  const SourceLocation SemicolonPos = utils::lexer::findNextTerminator(
      VoidReturn->getEndLoc(), *Result.SourceManager, getLangOpts());
  if (SemicolonPos.isInvalid())
    return;
  if (!SurroundingBlock) {
    const auto BraceInsertionHints = utils::getBraceInsertionsHints(
        VoidReturn, getLangOpts(), *Result.SourceManager,
        VoidReturn->getBeginLoc());
    if (BraceInsertionHints)
      Diag << BraceInsertionHints.openingBraceFixIt()
           << BraceInsertionHints.closingBraceFixIt();
  }
  Diag << FixItHint::CreateRemoval(VoidReturn->getReturnLoc());
  const auto *FunctionParent =
      Result.Nodes.getNodeAs<FunctionDecl>("function_parent");
  if (!FunctionParent ||
      (SurroundingBlock && SurroundingBlock->body_back() != VoidReturn))
    // If this is not the last statement in a function body, we add a `return`.
    Diag << FixItHint::CreateInsertion(SemicolonPos.getLocWithOffset(1),
                                       " return;", true);
}

void AvoidReturnWithVoidValueCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, IgnoreMacrosName, IgnoreMacros);
  Options.store(Opts, StrictModeName, StrictMode);
}

} // namespace clang::tidy::readability
