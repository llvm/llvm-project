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

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

LoopVariableCopiedThenModifiedCheck::LoopVariableCopiedThenModifiedCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreInexpensiveVariables(
          Options.get("IgnoreInexpensiveVariables", false)) {}

void LoopVariableCopiedThenModifiedCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreInexpensiveVariables", IgnoreInexpensiveVariables);
}

void LoopVariableCopiedThenModifiedCheck::registerMatchers(
    MatchFinder *Finder) {
  const auto HasReferenceOrPointerTypeOrIsAllowed = hasType(qualType(
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
      varDecl(HasReferenceOrPointerTypeOrIsAllowed,
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
  if (LoopVar->getBeginLoc().isMacroID())
    return;
  std::optional<bool> Expensive = utils::type_traits::isExpensiveToCopy(
      LoopVar->getType(), *Result.Context);
  if ((!Expensive || !*Expensive) && IgnoreInexpensiveVariables)
    return;
  const auto *ForRange = Result.Nodes.getNodeAs<CXXForRangeStmt>("forRange");

  std::string HintString = "";

  if (ExprMutationAnalyzer(*ForRange->getBody(), *Result.Context)
          .isMutated(LoopVar)) {
    if (isa<AutoType>(LoopVar->getType())) {
      HintString = "const auto&";
    } else {
      const std::string CanonicalTypeStr =
          LoopVar->getType().getAsString(Result.Context->getLangOpts());
      HintString = "const " + CanonicalTypeStr + "&";
    }
    clang::SourceRange LoopVarSourceRange =
        LoopVar->getTypeSourceInfo()->getTypeLoc().getSourceRange();
    diag(LoopVar->getLocation(), "loop variable '%0' is copied and then "
                                 "modified, which is likely a bug; you "
                                 "probably want to modify the underlying "
                                 "object and not this copy. If you "
                                 "*did* intend to modify this copy, "
                                 "please use an explicit copy inside the "
                                 "body of the loop")
        << LoopVar->getName()
        << FixItHint::CreateReplacement(LoopVarSourceRange, HintString);
  }
}

} // namespace clang::tidy::bugprone
