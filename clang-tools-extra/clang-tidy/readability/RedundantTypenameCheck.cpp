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
#include "clang/Sema/DeclSpec.h"

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;

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
          parmVarDecl(hasParent(typeLoc(hasParent(
              decl(anyOf(cxxMethodDecl(), hasParent(friendDecl()),
                         functionDecl(has(nestedNameSpecifier())))))))),
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
      if (const auto TTL = NonDependentTypeLoc->getAs<TypedefTypeLoc>())
        return TTL.getElaboratedKeywordLoc();

      if (const auto TTL = NonDependentTypeLoc->getAs<TagTypeLoc>())
        return TTL.getElaboratedKeywordLoc();
    } else {
      TypeLoc InnermostTypeLoc =
          *Result.Nodes.getNodeAs<TypeLoc>("dependentTypeLoc");
      while (const TypeLoc Next = InnermostTypeLoc.getNextTypeLoc())
        InnermostTypeLoc = Next;

      if (const auto DNTL = InnermostTypeLoc.getAs<DependentNameTypeLoc>())
        return DNTL.getElaboratedKeywordLoc();

      if (const auto TSTL =
              InnermostTypeLoc.getAs<TemplateSpecializationTypeLoc>())
        return TSTL.getElaboratedKeywordLoc();
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
