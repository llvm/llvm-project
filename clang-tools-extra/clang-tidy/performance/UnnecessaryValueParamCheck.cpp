//===--- UnnecessaryValueParamCheck.cpp - clang-tidy-----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnnecessaryValueParamCheck.h"

#include "../utils/DeclRefExprUtils.h"
#include "../utils/FixItHintUtils.h"
#include "../utils/Matchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace performance {

namespace {

std::string paramNameOrIndex(StringRef Name, size_t Index) {
  return (Name.empty() ? llvm::Twine('#') + llvm::Twine(Index + 1)
                       : llvm::Twine('\'') + Name + llvm::Twine('\''))
      .str();
}

} // namespace

void UnnecessaryValueParamCheck::registerMatchers(MatchFinder *Finder) {
  const auto ExpensiveValueParamDecl =
      parmVarDecl(hasType(hasCanonicalType(allOf(matchers::isExpensiveToCopy(),
                                                 unless(referenceType())))),
                  decl().bind("param"));
  Finder->addMatcher(
      functionDecl(isDefinition(), unless(cxxMethodDecl(isOverride())),
                   unless(isInstantiated()),
                   has(typeLoc(forEach(ExpensiveValueParamDecl))),
                   decl().bind("functionDecl")),
      this);
}

void UnnecessaryValueParamCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("functionDecl");
  const size_t Index = std::find(Function->parameters().begin(),
                                 Function->parameters().end(), Param) -
                       Function->parameters().begin();
  bool IsConstQualified =
      Param->getType().getCanonicalType().isConstQualified();

  // Skip declarations delayed by late template parsing without a body.
  if (!Function->getBody())
    return;

  // Do not trigger on non-const value parameters when:
  // 1. they are in a constructor definition since they can likely trigger
  //    misc-move-constructor-init which will suggest to move the argument.
  // 2. they are not only used as const.
  if (!IsConstQualified && (llvm::isa<CXXConstructorDecl>(Function) ||
                            !Function->doesThisDeclarationHaveABody() ||
                            !utils::decl_ref_expr::isOnlyUsedAsConst(
                                *Param, *Function->getBody(), *Result.Context)))
    return;
  auto Diag =
      diag(Param->getLocation(),
           IsConstQualified ? "the const qualified parameter %0 is "
                              "copied for each invocation; consider "
                              "making it a reference"
                            : "the parameter %0 is copied for each "
                              "invocation but only used as a const reference; "
                              "consider making it a const reference")
      << paramNameOrIndex(Param->getName(), Index);
  // Do not propose fixes in macros since we cannot place them correctly.
  if (Param->getLocStart().isMacroID())
    return;
  for (const auto *FunctionDecl = Function; FunctionDecl != nullptr;
       FunctionDecl = FunctionDecl->getPreviousDecl()) {
    const auto &CurrentParam = *FunctionDecl->getParamDecl(Index);
    Diag << utils::fixit::changeVarDeclToReference(CurrentParam,
                                                           *Result.Context);
    if (!IsConstQualified)
      Diag << utils::fixit::changeVarDeclToConst(CurrentParam);
  }
}

} // namespace performance
} // namespace tidy
} // namespace clang
