//===--- ForRangeCopyCheck.cpp - clang-tidy--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ForRangeCopyCheck.h"
#include "../utils/DeclRefExprUtils.h"
#include "../utils/FixItHintUtils.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "../utils/TypeTraits.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"
#include "clang/Basic/Diagnostic.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace performance {

ForRangeCopyCheck::ForRangeCopyCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WarnOnAllAutoCopies(Options.get("WarnOnAllAutoCopies", false)),
      AllowedTypes(
          utils::options::parseStringList(Options.get("AllowedTypes", ""))) {}

void ForRangeCopyCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WarnOnAllAutoCopies", WarnOnAllAutoCopies);
  Options.store(Opts, "AllowedTypes",
                utils::options::serializeStringList(AllowedTypes));
}

void ForRangeCopyCheck::registerMatchers(MatchFinder *Finder) {
  // Match loop variables that are not references or pointers or are already
  // initialized through MaterializeTemporaryExpr which indicates a type
  // conversion.
  auto HasReferenceOrPointerTypeOrIsAllowed = hasType(qualType(
      unless(anyOf(hasCanonicalType(anyOf(referenceType(), pointerType())),
                   hasDeclaration(namedDecl(
                       matchers::matchesAnyListedName(AllowedTypes)))))));
  auto IteratorReturnsValueType = cxxOperatorCallExpr(
      hasOverloadedOperatorName("*"),
      callee(
          cxxMethodDecl(returns(unless(hasCanonicalType(referenceType()))))));
  auto LoopVar =
      varDecl(HasReferenceOrPointerTypeOrIsAllowed,
              unless(hasInitializer(expr(hasDescendant(expr(anyOf(
                  materializeTemporaryExpr(), IteratorReturnsValueType)))))));
  Finder->addMatcher(
      traverse(ast_type_traits::TK_AsIs,
               cxxForRangeStmt(hasLoopVariable(LoopVar.bind("loopVar")))
                   .bind("forRange")),
      this);
}

void ForRangeCopyCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Var = Result.Nodes.getNodeAs<VarDecl>("loopVar");

  // Ignore code in macros since we can't place the fixes correctly.
  if (Var->getBeginLoc().isMacroID())
    return;
  if (handleConstValueCopy(*Var, *Result.Context))
    return;
  const auto *ForRange = Result.Nodes.getNodeAs<CXXForRangeStmt>("forRange");
  handleCopyIsOnlyConstReferenced(*Var, *ForRange, *Result.Context);
}

bool ForRangeCopyCheck::handleConstValueCopy(const VarDecl &LoopVar,
                                             ASTContext &Context) {
  if (WarnOnAllAutoCopies) {
    // For aggressive check just test that loop variable has auto type.
    if (!isa<AutoType>(LoopVar.getType()))
      return false;
  } else if (!LoopVar.getType().isConstQualified()) {
    return false;
  }
  llvm::Optional<bool> Expensive =
      utils::type_traits::isExpensiveToCopy(LoopVar.getType(), Context);
  if (!Expensive || !*Expensive)
    return false;
  auto Diagnostic =
      diag(LoopVar.getLocation(),
           "the loop variable's type is not a reference type; this creates a "
           "copy in each iteration; consider making this a reference")
      << utils::fixit::changeVarDeclToReference(LoopVar, Context);
  if (!LoopVar.getType().isConstQualified()) {
    if (llvm::Optional<FixItHint> Fix = utils::fixit::addQualifierToVarDecl(
            LoopVar, Context, DeclSpec::TQ::TQ_const))
      Diagnostic << *Fix;
  }
  return true;
}

bool ForRangeCopyCheck::handleCopyIsOnlyConstReferenced(
    const VarDecl &LoopVar, const CXXForRangeStmt &ForRange,
    ASTContext &Context) {
  llvm::Optional<bool> Expensive =
      utils::type_traits::isExpensiveToCopy(LoopVar.getType(), Context);
  if (LoopVar.getType().isConstQualified() || !Expensive || !*Expensive)
    return false;
  // We omit the case where the loop variable is not used in the loop body. E.g.
  //
  // for (auto _ : benchmark_state) {
  // }
  //
  // Because the fix (changing to `const auto &`) will introduce an unused
  // compiler warning which can't be suppressed.
  // Since this case is very rare, it is safe to ignore it.
  if (!ExprMutationAnalyzer(*ForRange.getBody(), Context).isMutated(&LoopVar) &&
      !utils::decl_ref_expr::allDeclRefExprs(LoopVar, *ForRange.getBody(),
                                             Context)
           .empty()) {
    auto Diag = diag(
        LoopVar.getLocation(),
        "loop variable is copied but only used as const reference; consider "
        "making it a const reference");

    if (llvm::Optional<FixItHint> Fix = utils::fixit::addQualifierToVarDecl(
            LoopVar, Context, DeclSpec::TQ::TQ_const))
      Diag << *Fix << utils::fixit::changeVarDeclToReference(LoopVar, Context);

    return true;
  }
  return false;
}

} // namespace performance
} // namespace tidy
} // namespace clang
