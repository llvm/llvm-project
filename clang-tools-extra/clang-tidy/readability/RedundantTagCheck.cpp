//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantTagCheck.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static bool canHideTag(const NamedDecl *D) {
  D = D->getUnderlyingDecl();

  return isa<VarDecl>(D) || isa<EnumConstantDecl>(D) || isa<FunctionDecl>(D) ||
         isa<FunctionTemplateDecl>(D) || isa<FieldDecl>(D) ||
         isa<UnresolvedUsingValueDecl>(D);
}

void RedundantTagCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      typeLoc(unless(hasAncestor(decl(isInstantiated())))).bind("typeLoc"),
      this);
}

void RedundantTagCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *TL = Result.Nodes.getNodeAs<TypeLoc>("typeLoc");
  if (!TL)
    return;

  if (TL->getType()->isInstantiationDependentType())
    return;

  const auto TagTL = TL->getAs<TagTypeLoc>();
  if (!TagTL)
    return;

  const TagDecl *TD = TagTL.getDecl();
  if (!TD)
    return;

  auto Lookup = TD->getDeclContext()->lookup(TD->getDeclName());

  for (const NamedDecl *ND : Lookup) {
    if (declaresSameEntity(ND, TD))
      continue;

    if (canHideTag(ND))
      return;
  }

  const SourceLocation KeywordLoc = TagTL.getElaboratedKeywordLoc();
  if (KeywordLoc.isInvalid())
    return;

  Token Tok;
  if (Lexer::getRawToken(KeywordLoc, Tok, *Result.SourceManager, getLangOpts()))
    return;

  const llvm::StringRef Keyword = Tok.getRawIdentifier();

  if (Keyword != "struct" && Keyword != "class" && Keyword != "union" &&
      Keyword != "enum")
    return;

  diag(KeywordLoc, "redundant '%0' keyword in C++ declaration")
      << Keyword << FixItHint::CreateRemoval(KeywordLoc);
}

} // namespace clang::tidy::readability
