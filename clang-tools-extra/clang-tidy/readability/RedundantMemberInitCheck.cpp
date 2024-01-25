//===--- RedundantMemberInitCheck.cpp - clang-tidy-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantMemberInitCheck.h"
#include "../utils/LexerUtils.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include <algorithm>

using namespace clang::ast_matchers;
using namespace clang::tidy::matchers;

namespace clang::tidy::readability {

static SourceRange
getFullInitRangeInclWhitespaces(SourceRange Range, const SourceManager &SM,
                                const LangOptions &LangOpts) {
  const Token PrevToken =
      utils::lexer::getPreviousToken(Range.getBegin(), SM, LangOpts, false);
  if (PrevToken.is(tok::unknown))
    return Range;

  if (PrevToken.isNot(tok::equal))
    return {PrevToken.getEndLoc(), Range.getEnd()};

  return getFullInitRangeInclWhitespaces(
      {PrevToken.getLocation(), Range.getEnd()}, SM, LangOpts);
}

void RedundantMemberInitCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreBaseInCopyConstructors",
                IgnoreBaseInCopyConstructors);
}

void RedundantMemberInitCheck::registerMatchers(MatchFinder *Finder) {
  auto ConstructorMatcher =
      cxxConstructExpr(argumentCountIs(0),
                       hasDeclaration(cxxConstructorDecl(ofClass(cxxRecordDecl(
                           unless(isTriviallyDefaultConstructible()))))))
          .bind("construct");

  Finder->addMatcher(
      cxxConstructorDecl(
          unless(isDelegatingConstructor()), ofClass(unless(isUnion())),
          forEachConstructorInitializer(
              cxxCtorInitializer(withInitializer(ConstructorMatcher),
                                 unless(forField(fieldDecl(
                                     anyOf(hasType(isConstQualified()),
                                           hasParent(recordDecl(isUnion())))))))
                  .bind("init")))
          .bind("constructor"),
      this);

  Finder->addMatcher(fieldDecl(hasInClassInitializer(ConstructorMatcher),
                               unless(hasParent(recordDecl(isUnion()))))
                         .bind("field"),
                     this);
}

void RedundantMemberInitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Construct = Result.Nodes.getNodeAs<CXXConstructExpr>("construct");

  if (const auto *Field = Result.Nodes.getNodeAs<FieldDecl>("field")) {
    const Expr *Init = Field->getInClassInitializer();
    diag(Construct->getExprLoc(), "initializer for member %0 is redundant")
        << Field
        << FixItHint::CreateRemoval(getFullInitRangeInclWhitespaces(
               Init->getSourceRange(), *Result.SourceManager, getLangOpts()));
    return;
  }

  const auto *Init = Result.Nodes.getNodeAs<CXXCtorInitializer>("init");
  const auto *ConstructorDecl =
      Result.Nodes.getNodeAs<CXXConstructorDecl>("constructor");

  if (IgnoreBaseInCopyConstructors && ConstructorDecl->isCopyConstructor() &&
      Init->isBaseInitializer())
    return;

  if (Init->isAnyMemberInitializer()) {
    diag(Init->getSourceLocation(), "initializer for member %0 is redundant")
        << Init->getAnyMember()
        << FixItHint::CreateRemoval(Init->getSourceRange());
  } else {
    diag(Init->getSourceLocation(),
         "initializer for base class %0 is redundant")
        << Construct->getType()
        << FixItHint::CreateRemoval(Init->getSourceRange());
  }
}

} // namespace clang::tidy::readability
