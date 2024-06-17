//===--- PreferAtOverSubscriptOperatorCheck.cpp - clang-tidy --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PreferAtOverSubscriptOperatorCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/StringRef.h"
#include <numeric>

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

static constexpr std::array<llvm::StringRef, 3> DefaultExclusions = {
    llvm::StringRef("::std::map"), llvm::StringRef("::std::unordered_map"),
    llvm::StringRef("::std::flat_map")};

PreferAtOverSubscriptOperatorCheck::PreferAtOverSubscriptOperatorCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {

  ExcludedClasses = clang::tidy::utils::options::parseStringList(
      Options.get("ExcludeClasses", ""));
  ExcludedClasses.insert(ExcludedClasses.end(), DefaultExclusions.begin(),
                         DefaultExclusions.end());
}

void PreferAtOverSubscriptOperatorCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {

  if (ExcludedClasses.size() == DefaultExclusions.size()) {
    Options.store(Opts, "ExcludeClasses", "");
    return;
  }

  // Sum up the sizes of the defaults ( + semicolons), so we can remove them
  // from the saved options
  size_t DefaultsStringLength =
      std::transform_reduce(DefaultExclusions.begin(), DefaultExclusions.end(),
                            DefaultExclusions.size(), std::plus<>(),
                            [](llvm::StringRef Name) { return Name.size(); });

  std::string Serialized =
      clang::tidy::utils::options::serializeStringList(ExcludedClasses);

  Options.store(Opts, "ExcludeClasses",
                Serialized.substr(0, Serialized.size() - DefaultsStringLength));
}

const CXXMethodDecl *findAlternative(const CXXRecordDecl *MatchedParent,
                                     const CXXMethodDecl *MatchedOperator) {
  for (const CXXMethodDecl *Method : MatchedParent->methods()) {
    const bool CorrectName = Method->getNameInfo().getAsString() == "at";
    if (!CorrectName)
      continue;

    const bool SameReturnType =
        Method->getReturnType() == MatchedOperator->getReturnType();
    if (!SameReturnType)
      continue;

    const bool SameNumberOfArguments =
        Method->getNumParams() == MatchedOperator->getNumParams();
    if (!SameNumberOfArguments)
      continue;

    for (unsigned ArgInd = 0; ArgInd < Method->getNumParams(); ArgInd++) {
      const bool SameArgType =
          Method->parameters()[ArgInd]->getOriginalType() ==
          MatchedOperator->parameters()[ArgInd]->getOriginalType();
      if (!SameArgType)
        continue;
    }

    return Method;
  }
  return static_cast<CXXMethodDecl *>(nullptr);
}

void PreferAtOverSubscriptOperatorCheck::registerMatchers(MatchFinder *Finder) {
  // Need a callExpr here to match CXXOperatorCallExpr ``(&a)->operator[](0)``
  // and CXXMemberCallExpr ``a[0]``.
  Finder->addMatcher(
      callExpr(
          callee(
              cxxMethodDecl(hasOverloadedOperatorName("[]")).bind("operator")),
          callee(cxxMethodDecl(
              ofClass(cxxRecordDecl(hasMethod(hasName("at"))).bind("parent")),
              unless(matchers::matchesAnyListedName(ExcludedClasses)))))
          .bind("caller"),
      this);
}

void PreferAtOverSubscriptOperatorCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<CallExpr>("caller");
  const auto *MatchedOperator =
      Result.Nodes.getNodeAs<CXXMethodDecl>("operator");
  const auto *MatchedParent = Result.Nodes.getNodeAs<CXXRecordDecl>("parent");

  const CXXMethodDecl *Alternative =
      findAlternative(MatchedParent, MatchedOperator);
  if (!Alternative)
    return;

  diag(MatchedExpr->getBeginLoc(),
       "found possibly unsafe operator[], consider using at() instead")
      << MatchedExpr->getSourceRange();
  diag(Alternative->getBeginLoc(), "alternative at() defined here",
       DiagnosticIDs::Note)
      << Alternative->getSourceRange();
}

} // namespace clang::tidy::cppcoreguidelines
