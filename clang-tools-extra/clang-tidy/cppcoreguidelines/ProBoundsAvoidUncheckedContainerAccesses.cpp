//===--- ProBoundsAvoidUncheckedContainerAccesses.cpp - clang-tidy --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProBoundsAvoidUncheckedContainerAccesses.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/StringRef.h"
#include <numeric>

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

static constexpr std::array<llvm::StringRef, 3> SubscriptDefaultExclusions = {
    llvm::StringRef("::std::map"), llvm::StringRef("::std::unordered_map"),
    llvm::StringRef("::std::flat_map")};

ProBoundsAvoidUncheckedContainerAccesses::
    ProBoundsAvoidUncheckedContainerAccesses(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {

  SubscriptExcludedClasses = clang::tidy::utils::options::parseStringList(
      Options.get("ExcludeClasses", ""));
  SubscriptExcludedClasses.insert(SubscriptExcludedClasses.end(),
                                  SubscriptDefaultExclusions.begin(),
                                  SubscriptDefaultExclusions.end());
  SubscriptFixMode = Options.get("SubscriptFixMode", None);
  SubscriptFixFunction = Options.get("SubscriptFixFunction", "gsl::at");
}

void ProBoundsAvoidUncheckedContainerAccesses::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {

  Options.store(Opts, "SubscriptFixFunction", SubscriptFixFunction);
  Options.store(Opts, "SubscriptFixMode", SubscriptFixMode);
  if (SubscriptExcludedClasses.size() == SubscriptDefaultExclusions.size()) {
    Options.store(Opts, "ExcludeClasses", "");
    return;
  }

  // Sum up the sizes of the defaults ( + semicolons), so we can remove them
  // from the saved options
  size_t DefaultsStringLength = std::transform_reduce(
      SubscriptDefaultExclusions.begin(), SubscriptDefaultExclusions.end(),
      SubscriptDefaultExclusions.size(), std::plus<>(),
      [](llvm::StringRef Name) { return Name.size(); });

  std::string Serialized = clang::tidy::utils::options::serializeStringList(
      SubscriptExcludedClasses);

  Options.store(Opts, "SubscriptExcludeClasses",
                Serialized.substr(0, Serialized.size() - DefaultsStringLength));
}

// TODO: if at() is defined in another class in the class hierarchy of the class
// that defines the operator[] we matched on, findAlternative() will not detect
// it.
static const CXXMethodDecl *
findAlternativeAt(const CXXMethodDecl *MatchedOperator) {
  const CXXRecordDecl *Parent = MatchedOperator->getParent();
  const QualType SubscriptThisObjType =
      MatchedOperator->getFunctionObjectParameterReferenceType();

  for (const CXXMethodDecl *Method : Parent->methods()) {
    // Require 'Method' to be as accessible as 'MatchedOperator' or more
    if (MatchedOperator->getAccess() < Method->getAccess())
      continue;

    if (MatchedOperator->isConst() != Method->isConst())
      continue;

    const QualType AtThisObjType =
        Method->getFunctionObjectParameterReferenceType();
    if (SubscriptThisObjType != AtThisObjType)
      continue;

    if (!Method->getNameInfo().getName().isIdentifier() ||
        Method->getName() != "at")
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
  return nullptr;
}

void ProBoundsAvoidUncheckedContainerAccesses::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(
      mapAnyOf(cxxOperatorCallExpr, cxxMemberCallExpr)
          .with(callee(cxxMethodDecl(hasOverloadedOperatorName("[]"),
                                     unless(matchers::matchesAnyListedName(
                                         SubscriptExcludedClasses)))
                           .bind("operator")))
          .bind("caller"),
      this);
}

void ProBoundsAvoidUncheckedContainerAccesses::check(
    const MatchFinder::MatchResult &Result) {

  const auto *MatchedExpr = Result.Nodes.getNodeAs<CallExpr>("caller");

  if (SubscriptFixMode == None) {
    diag(MatchedExpr->getCallee()->getBeginLoc(),
         "possibly unsafe 'operator[]', consider bounds-safe alternatives")
        << MatchedExpr->getCallee()->getSourceRange();
    return;
  }

  if (const auto *OCE = dyn_cast<CXXOperatorCallExpr>(MatchedExpr)) {
    // Case: a[i]
    auto LeftBracket = SourceRange(OCE->getCallee()->getBeginLoc(),
                                   OCE->getCallee()->getBeginLoc());
    auto RightBracket =
        SourceRange(OCE->getOperatorLoc(), OCE->getOperatorLoc());

    if (SubscriptFixMode == At) {
      // Case: a[i] => a.at(i)
      const auto *MatchedOperator =
          Result.Nodes.getNodeAs<CXXMethodDecl>("operator");
      const CXXMethodDecl *Alternative = findAlternativeAt(MatchedOperator);

      if (!Alternative) {
        diag(MatchedExpr->getCallee()->getBeginLoc(),
             "possibly unsafe 'operator[]', consider "
             "bounds-safe alternatives")
            << MatchedExpr->getCallee()->getSourceRange();
        return;
      }

      diag(MatchedExpr->getCallee()->getBeginLoc(),
           "possibly unsafe 'operator[]', consider "
           "bounds-safe alternative 'at()'")
          << MatchedExpr->getCallee()->getSourceRange()
          << FixItHint::CreateReplacement(LeftBracket, ".at(")
          << FixItHint::CreateReplacement(RightBracket, ")");

      diag(Alternative->getBeginLoc(), "viable 'at()' is defined here",
           DiagnosticIDs::Note)
          << Alternative->getNameInfo().getSourceRange();

    } else if (SubscriptFixMode == Function) {
      // Case: a[i] => f(a, i)
      diag(MatchedExpr->getCallee()->getBeginLoc(),
           "possibly unsafe 'operator[]', use safe function '" +
               SubscriptFixFunction.str() + "()' instead")
          << MatchedExpr->getCallee()->getSourceRange()
          << FixItHint::CreateInsertion(MatchedExpr->getBeginLoc(),
                                        SubscriptFixFunction.str() + "(")
          // Since C++23, the subscript operator may also be called without an
          // argument, which makes the following distinction necessary
          << (MatchedExpr->getDirectCallee()->getNumParams() > 0
                  ? FixItHint::CreateReplacement(LeftBracket, ", ")
                  : FixItHint::CreateRemoval(LeftBracket))
          << FixItHint::CreateReplacement(RightBracket, ")");
    }
  } else if (const auto *MCE = dyn_cast<CXXMemberCallExpr>(MatchedExpr)) {
    // Case: a.operator[](i) or a->operator[](i)
    const auto *Callee = dyn_cast<MemberExpr>(MCE->getCallee());

    if (SubscriptFixMode == At) {
      // Cases: a.operator[](i) => a.at(i) and a->operator[](i) => a->at(i)

      const auto *MatchedOperator =
          Result.Nodes.getNodeAs<CXXMethodDecl>("operator");

      const CXXMethodDecl *Alternative = findAlternativeAt(MatchedOperator);
      if (!Alternative) {
        diag(Callee->getBeginLoc(), "possibly unsafe 'operator[]', consider "
                                    "bounds-safe alternative 'at()'")
            << Callee->getSourceRange();
        return;
      }
      diag(MatchedExpr->getCallee()->getBeginLoc(),
           "possibly unsafe 'operator[]', consider "
           "bounds-safe alternative 'at()'")
          << FixItHint::CreateReplacement(
                 SourceRange(Callee->getMemberLoc(), Callee->getEndLoc()),
                 "at");

      diag(Alternative->getBeginLoc(), "viable 'at()' defined here",
           DiagnosticIDs::Note)
          << Alternative->getNameInfo().getSourceRange();

    } else if (SubscriptFixMode == Function) {
      // Cases: a.operator[](i) => f(a, i) and a->operator[](i) => f(*a, i)
      const auto *Callee = dyn_cast<MemberExpr>(MCE->getCallee());
      std::string BeginInsertion = SubscriptFixFunction.str() + "(";

      if (Callee->isArrow())
        BeginInsertion += "*";

      diag(Callee->getBeginLoc(),
           "possibly unsafe 'operator[]', use safe function '" +
               SubscriptFixFunction.str() + "()' instead")
          << Callee->getSourceRange()
          << FixItHint::CreateInsertion(MatchedExpr->getBeginLoc(),
                                        BeginInsertion)
          // Since C++23, the subscript operator may also be called without an
          // argument, which makes the following distinction necessary
          << ((MCE->getMethodDecl()->getNumNonObjectParams() > 0)
                  ? FixItHint::CreateReplacement(
                        SourceRange(
                            Callee->getOperatorLoc(),
                            MCE->getArg(0)->getBeginLoc().getLocWithOffset(-1)),
                        ", ")
                  : FixItHint::CreateRemoval(
                        SourceRange(Callee->getOperatorLoc(),
                                    MCE->getRParenLoc().getLocWithOffset(-1))));
    }
  }
}

} // namespace clang::tidy::cppcoreguidelines

namespace clang::tidy {
using P = cppcoreguidelines::ProBoundsAvoidUncheckedContainerAccesses;

llvm::ArrayRef<std::pair<P::SubscriptFixModes, StringRef>>
OptionEnumMapping<P::SubscriptFixModes>::getEnumMapping() {
  static constexpr std::pair<P::SubscriptFixModes, StringRef> Mapping[] = {
      {P::None, "None"}, {P::At, "at"}, {P::Function, "function"}};
  return {Mapping};
}
} // namespace clang::tidy
