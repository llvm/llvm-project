
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
  auto HasReferenceOrPointerTypeOrIsAllowed = hasType(qualType(
      unless(hasCanonicalType(anyOf(referenceType(), pointerType())))));
  auto IteratorReturnsValueType = cxxOperatorCallExpr(
      hasOverloadedOperatorName("*"),
      callee(
          cxxMethodDecl(returns(unless(hasCanonicalType(referenceType()))))));
  auto NotConstructedByCopy = cxxConstructExpr(
      hasDeclaration(cxxConstructorDecl(unless(isCopyConstructor()))));
  auto ConstructedByConversion = cxxMemberCallExpr(callee(cxxConversionDecl()));
  auto LoopVar =
      varDecl(HasReferenceOrPointerTypeOrIsAllowed,
              unless(hasInitializer(expr(hasDescendant(expr(
                  anyOf(materializeTemporaryExpr(), IteratorReturnsValueType,
                        NotConstructedByCopy, ConstructedByConversion)))))));
  Finder->addMatcher(
      traverse(TK_AsIs,
               cxxForRangeStmt(hasLoopVariable(LoopVar.bind("loopVar")))
                   .bind("forRange")),
      this);
}

void LoopVariableCopiedThenModifiedCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Var = Result.Nodes.getNodeAs<VarDecl>("loopVar");
  if (Var->getBeginLoc().isMacroID())
    return;
  std::optional<bool> Expensive =
      utils::type_traits::isExpensiveToCopy(Var->getType(), *Result.Context);
  if ((!Expensive || !*Expensive) && IgnoreInexpensiveVariables)
    return;
  const auto *ForRange = Result.Nodes.getNodeAs<CXXForRangeStmt>("forRange");
  if (copiedLoopVarIsMutated(*Var, *ForRange, *Result.Context))
    return;
}

bool LoopVariableCopiedThenModifiedCheck::copiedLoopVarIsMutated(
    const VarDecl &LoopVar, const CXXForRangeStmt &ForRange,
    ASTContext &Context) {

  std::string hintstring = "";

  if (ExprMutationAnalyzer(*ForRange.getBody(), Context).isMutated(&LoopVar)) {
    if (isa<AutoType>(LoopVar.getType())) {
      hintstring = "const auto&";
    } else {
      std::string CanonicalTypeStr =
          LoopVar.getType().getAsString(Context.getLangOpts());
      hintstring = "const " + CanonicalTypeStr + "&";
    }
    clang::SourceRange loopvar_source_range =
        LoopVar.getTypeSourceInfo()->getTypeLoc().getSourceRange();
    auto Diag =
        diag(LoopVar.getLocation(), "loop variable '%0' is copied and then "
                                    "modified, which is likely a bug; you "
                                    "probably want to modify the underlying "
                                    "object and not this copy. If you "
                                    "*did* intend to modify this copy, "
                                    "please use an explicit copy inside the "
                                    "body of the loop")
        << LoopVar.getName()
        << FixItHint::CreateReplacement(loopvar_source_range, hintstring);
    return true;
  }
  return false;
}

} // namespace clang::tidy::bugprone
