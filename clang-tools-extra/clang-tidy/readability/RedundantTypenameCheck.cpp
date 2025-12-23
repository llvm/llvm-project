//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantTypenameCheck.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void RedundantTypenameCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(typeLoc(unless(hasAncestor(decl(isInstantiated()))))
                         .bind("nonDependentTypeLoc"),
                     this);

  if (!getLangOpts().CPlusPlus20)
    return;

  const auto InImplicitTypenameContext = anyOf(
      hasParent(decl(anyOf(
          typedefNameDecl(), templateTypeParmDecl(), nonTypeTemplateParmDecl(),
          friendDecl(), fieldDecl(),
          varDecl(hasDeclContext(anyOf(namespaceDecl(), translationUnitDecl())),
                  unless(parmVarDecl())),
          parmVarDecl(hasParent(expr(requiresExpr()))),
          parmVarDecl(hasParent(typeLoc(hasParent(decl(
              anyOf(cxxMethodDecl(), hasParent(friendDecl()),
                    functionDecl(has(nestedNameSpecifier())),
                    cxxDeductionGuideDecl(hasDeclContext(recordDecl())))))))),
          // Match return types.
          functionDecl(unless(cxxConversionDecl()))))),
      hasParent(expr(anyOf(cxxNamedCastExpr(), cxxNewExpr()))));
  Finder->addMatcher(
      typeLoc(InImplicitTypenameContext).bind("dependentTypeLoc"), this);
}

void RedundantTypenameCheck::check(const MatchFinder::MatchResult &Result) {
  const SourceLocation ElaboratedKeywordLoc = [&] {
    if (const auto *NonDependentTypeLoc =
            Result.Nodes.getNodeAs<TypeLoc>("nonDependentTypeLoc")) {
      if (const auto TL = NonDependentTypeLoc->getAs<TypedefTypeLoc>())
        return TL.getElaboratedKeywordLoc();

      if (const auto TL = NonDependentTypeLoc->getAs<TagTypeLoc>())
        return TL.getElaboratedKeywordLoc();

      if (const auto TL = NonDependentTypeLoc
                              ->getAs<DeducedTemplateSpecializationTypeLoc>())
        return TL.getElaboratedKeywordLoc();

      if (const auto TL =
              NonDependentTypeLoc->getAs<TemplateSpecializationTypeLoc>())
        if (!TL.getType()->isDependentType())
          return TL.getElaboratedKeywordLoc();
    } else {
      TypeLoc InnermostTypeLoc =
          *Result.Nodes.getNodeAs<TypeLoc>("dependentTypeLoc");
      while (const TypeLoc Next = InnermostTypeLoc.getNextTypeLoc())
        InnermostTypeLoc = Next;

      if (const auto TL = InnermostTypeLoc.getAs<DependentNameTypeLoc>())
        return TL.getElaboratedKeywordLoc();

      if (const auto TL =
              InnermostTypeLoc.getAs<TemplateSpecializationTypeLoc>())
        return TL.getElaboratedKeywordLoc();
    }

    return SourceLocation();
  }();

  if (ElaboratedKeywordLoc.isInvalid())
    return;

  if (Token ElaboratedKeyword;
      Lexer::getRawToken(ElaboratedKeywordLoc, ElaboratedKeyword,
                         *Result.SourceManager, getLangOpts()) ||
      ElaboratedKeyword.getRawIdentifier() != "typename")
    return;

  diag(ElaboratedKeywordLoc, "redundant 'typename'")
      << FixItHint::CreateRemoval(ElaboratedKeywordLoc);
}

} // namespace clang::tidy::readability
