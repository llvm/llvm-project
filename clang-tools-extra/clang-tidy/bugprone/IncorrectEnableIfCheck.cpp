//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncorrectEnableIfCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER_P(TemplateTypeParmDecl, hasUnnamedDefaultArgument,
              ast_matchers::internal::Matcher<TypeLoc>, InnerMatcher) {
  if (Node.getIdentifier() != nullptr || !Node.hasDefaultArgument() ||
      Node.getDefaultArgument().getArgument().isNull())
    return false;

  TypeLoc DefaultArgTypeLoc =
      Node.getDefaultArgument().getTypeSourceInfo()->getTypeLoc();
  return InnerMatcher.matches(DefaultArgTypeLoc, Finder, Builder);
}

} // namespace

void IncorrectEnableIfCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      templateTypeParmDecl(
          hasUnnamedDefaultArgument(templateSpecializationTypeLoc(
                                        loc(qualType(hasDeclaration(namedDecl(
                                            hasName("::std::enable_if"))))))
                                        .bind("enable_if_specialization")))
          .bind("enable_if"),
      this);
}

void IncorrectEnableIfCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *EnableIf =
      Result.Nodes.getNodeAs<TemplateTypeParmDecl>("enable_if");
  const auto *EnableIfSpecializationLoc =
      Result.Nodes.getNodeAs<TemplateSpecializationTypeLoc>(
          "enable_if_specialization");

  if (!EnableIf || !EnableIfSpecializationLoc)
    return;

  const SourceManager &SM = *Result.SourceManager;
  SourceLocation RAngleLoc =
      SM.getExpansionLoc(EnableIfSpecializationLoc->getRAngleLoc());

  auto Diag = diag(EnableIf->getBeginLoc(),
                   "incorrect std::enable_if usage detected; use "
                   "'typename std::enable_if<...>::type'");
  // FIXME: This should handle the enable_if specialization already having an
  // elaborated keyword.
  if (!getLangOpts().CPlusPlus20) {
    Diag << FixItHint::CreateInsertion(EnableIfSpecializationLoc->getBeginLoc(),
                                       "typename ");
  }
  Diag << FixItHint::CreateInsertion(RAngleLoc.getLocWithOffset(1), "::type");
}

} // namespace clang::tidy::bugprone
