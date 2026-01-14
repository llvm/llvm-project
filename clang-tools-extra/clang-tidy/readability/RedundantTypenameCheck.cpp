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
  Finder->addMatcher(
      typeLoc(unless(hasAncestor(decl(isInstantiated())))).bind("typeLoc"),
      this);

  if (!getLangOpts().CPlusPlus20)
    return;

  const auto InImplicitTypenameContext =
      anyOf(hasParent(decl(anyOf(
                typedefNameDecl(), templateTypeParmDecl(),
                nonTypeTemplateParmDecl(), friendDecl(), fieldDecl(),
                parmVarDecl(hasParent(expr(requiresExpr()))),
                parmVarDecl(hasParent(typeLoc(hasParent(decl(anyOf(
                    cxxMethodDecl(), hasParent(friendDecl()),
                    functionDecl(has(nestedNameSpecifier())),
                    cxxDeductionGuideDecl(hasDeclContext(recordDecl())))))))),
                // Match return types.
                functionDecl(unless(cxxConversionDecl()))))),
            hasParent(expr(anyOf(cxxNamedCastExpr(), cxxNewExpr()))));
  Finder->addMatcher(
      typeLoc(InImplicitTypenameContext).bind("dependentTypeLoc"), this);
  Finder->addMatcher(
      varDecl(hasDeclContext(anyOf(namespaceDecl(), translationUnitDecl(),
                                   cxxRecordDecl())),
              unless(parmVarDecl()),
              hasTypeLoc(typeLoc().bind("dependentTypeLoc"))),
      this);
}

void RedundantTypenameCheck::check(const MatchFinder::MatchResult &Result) {
  const TypeLoc TL = [&] {
    if (const auto *TL = Result.Nodes.getNodeAs<TypeLoc>("typeLoc"))
      return TL->getType()->isDependentType() ? TypeLoc() : *TL;

    auto TL = *Result.Nodes.getNodeAs<TypeLoc>("dependentTypeLoc");
    while (const TypeLoc Next = TL.getNextTypeLoc())
      TL = Next;
    return TL;
  }();

  if (TL.isNull())
    return;

  const SourceLocation ElaboratedKeywordLoc = [&] {
    if (const auto CastTL = TL.getAs<TypedefTypeLoc>())
      return CastTL.getElaboratedKeywordLoc();

    if (const auto CastTL = TL.getAs<TagTypeLoc>())
      return CastTL.getElaboratedKeywordLoc();

    if (const auto CastTL = TL.getAs<DeducedTemplateSpecializationTypeLoc>())
      return CastTL.getElaboratedKeywordLoc();

    if (const auto CastTL = TL.getAs<TemplateSpecializationTypeLoc>())
      return CastTL.getElaboratedKeywordLoc();

    if (const auto CastTL = TL.getAs<DependentNameTypeLoc>())
      return CastTL.getElaboratedKeywordLoc();

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
