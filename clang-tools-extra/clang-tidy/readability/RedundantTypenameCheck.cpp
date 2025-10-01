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
#include "clang/Sema/DeclSpec.h"

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;

namespace clang::tidy::readability {

void RedundantTypenameCheck::registerMatchers(MatchFinder *Finder) {
  // NOLINTNEXTLINE(readability-identifier-naming)
  const VariadicDynCastAllOfMatcher<TypeLoc, TypedefTypeLoc> typedefTypeLoc;
  Finder->addMatcher(typedefTypeLoc().bind("typedefTypeLoc"), this);

  if (!getLangOpts().CPlusPlus20)
    return;

  // NOLINTBEGIN(readability-identifier-naming)
  const VariadicDynCastAllOfMatcher<Stmt, CXXNamedCastExpr> cxxNamedCastExpr;
  const auto inImplicitTypenameContext = anyOf(
      hasParent(typedefNameDecl()), hasParent(templateTypeParmDecl()),
      hasParent(nonTypeTemplateParmDecl()), hasParent(cxxNamedCastExpr()),
      hasParent(cxxNewExpr()), hasParent(friendDecl()), hasParent(fieldDecl()),
      hasParent(parmVarDecl(hasParent(expr(requiresExpr())))),
      hasParent(parmVarDecl(hasParent(typeLoc(hasParent(
          namedDecl(anyOf(cxxMethodDecl(), hasParent(friendDecl()),
                          functionDecl(has(nestedNameSpecifier()))))))))),
      // Match return types.
      hasParent(functionDecl(unless(cxxConversionDecl()))));
  // NOLINTEND(readability-identifier-naming)
  Finder->addMatcher(typeLoc(inImplicitTypenameContext).bind("typeloc"), this);
}

void RedundantTypenameCheck::check(const MatchFinder::MatchResult &Result) {
  const SourceLocation TypenameKeywordLoc = [&] {
    if (const auto *TTL =
            Result.Nodes.getNodeAs<TypedefTypeLoc>("typedefTypeLoc"))
      return TTL->getElaboratedKeywordLoc();

    TypeLoc InnermostTypeLoc = *Result.Nodes.getNodeAs<TypeLoc>("typeloc");
    while (const TypeLoc Next = InnermostTypeLoc.getNextTypeLoc())
      InnermostTypeLoc = Next;

    if (const auto DNTL = InnermostTypeLoc.getAs<DependentNameTypeLoc>())
      return DNTL.getElaboratedKeywordLoc();

    return SourceLocation();
  }();

  if (!TypenameKeywordLoc.isValid())
    return;

  diag(TypenameKeywordLoc, "redundant 'typename'")
      << FixItHint::CreateRemoval(TypenameKeywordLoc);
}

} // namespace clang::tidy::readability
