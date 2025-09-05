//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoopVariableCopiedThenModifiedCheck.h"
#include "../utils/Matchers.h"
#include "../utils/TypeTraits.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {
AST_MATCHER(VarDecl, isInMacro) { return Node.getBeginLoc().isMacroID(); }
} // namespace

LoopVariableCopiedThenModifiedCheck::LoopVariableCopiedThenModifiedCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), IgnoreInexpensiveVariables(Options.get(
                                         "IgnoreInexpensiveVariables", false)),
      WarnOnlyOnAutoCopies(Options.get("WarnOnlyOnAutoCopies", false)) {}

void LoopVariableCopiedThenModifiedCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreInexpensiveVariables", IgnoreInexpensiveVariables);
  Options.store(Opts, "WarnOnlyOnAutoCopies", WarnOnlyOnAutoCopies);
}

void LoopVariableCopiedThenModifiedCheck::registerMatchers(
    MatchFinder *Finder) {
  const auto HasReferenceOrPointerType = hasType(qualType(
      unless(hasCanonicalType(anyOf(referenceType(), pointerType())))));
  const auto IteratorReturnsValueType = cxxOperatorCallExpr(
      hasOverloadedOperatorName("*"),
      callee(
          cxxMethodDecl(returns(unless(hasCanonicalType(referenceType()))))));
  const auto NotConstructedByCopy = cxxConstructExpr(
      hasDeclaration(cxxConstructorDecl(unless(isCopyConstructor()))));
  const auto ConstructedByConversion =
      cxxMemberCallExpr(callee(cxxConversionDecl()));
  const auto LoopVar =
      varDecl(unless(isInMacro()), HasReferenceOrPointerType,
              unless(hasInitializer(expr(hasDescendant(expr(
                  anyOf(materializeTemporaryExpr(), IteratorReturnsValueType,
                        NotConstructedByCopy, ConstructedByConversion)))))));
  Finder->addMatcher(cxxForRangeStmt(hasLoopVariable(LoopVar.bind("loopVar")))
                         .bind("forRange"),
                     this);
}

void LoopVariableCopiedThenModifiedCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *LoopVar = Result.Nodes.getNodeAs<VarDecl>("loopVar");
  std::optional<bool> Expensive = utils::type_traits::isExpensiveToCopy(
      LoopVar->getType(), *Result.Context);
  if ((!Expensive || !*Expensive) && IgnoreInexpensiveVariables)
    return;
  if (WarnOnlyOnAutoCopies) {
    if (!isa<AutoType>(LoopVar->getType())) {
      return;
    }
  }
  const auto *ForRange = Result.Nodes.getNodeAs<CXXForRangeStmt>("forRange");

  if (!ExprMutationAnalyzer(*ForRange->getBody(), *Result.Context)
           .isMutated(LoopVar)) {
    return;
  }

  clang::SourceRange LoopVarSourceRange =
      LoopVar->getTypeSourceInfo()->getTypeLoc().getSourceRange();
  clang::SourceLocation EndLoc = clang::Lexer::getLocForEndOfToken(
      LoopVarSourceRange.getEnd(), 0, Result.Context->getSourceManager(),
      Result.Context->getLangOpts());
  diag(LoopVar->getLocation(),
       "loop variable '%0' is copied and then (possibly) modified; use an "
       "explicit copy inside the body of the loop or make the variable a "
       "reference")
      << LoopVar->getName();
  diag(LoopVar->getLocation(), "consider making '%0' a reference",
       DiagnosticIDs::Note)
      << LoopVar->getName()
      << FixItHint::CreateInsertion(LoopVarSourceRange.getBegin(), "const ")
      << FixItHint::CreateInsertion(EndLoc, "&");
}

} // namespace clang::tidy::bugprone
