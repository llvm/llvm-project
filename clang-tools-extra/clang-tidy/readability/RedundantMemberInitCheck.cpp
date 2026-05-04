//===----------------------------------------------------------------------===//
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

using namespace clang::ast_matchers;
using namespace clang::tidy::matchers;

namespace clang::tidy::readability {

static SourceRange
getFullInitRangeInclWhitespaces(SourceRange Range, const SourceManager &SM,
                                const LangOptions &LangOpts) {
  const std::optional<Token> PrevToken =
      utils::lexer::getPreviousToken(Range.getBegin(), SM, LangOpts, false);
  if (!PrevToken)
    return Range;

  if (PrevToken->isNot(tok::equal))
    return {PrevToken->getEndLoc(), Range.getEnd()};

  return getFullInitRangeInclWhitespaces(
      {PrevToken->getLocation(), Range.getEnd()}, SM, LangOpts);
}

namespace {
// Matches a ``CXXConstructExpr`` whose written argument list (i.e. the
// source text between the parentheses or braces) involves a macro.
AST_MATCHER(CXXConstructExpr, initListContainsMacro) {
  const SourceRange InitRange = Node.getParenOrBraceRange();
  if (InitRange.isInvalid())
    return false;
  if (InitRange.getBegin().isMacroID() || InitRange.getEnd().isMacroID())
    return true;
  const ASTContext &Context = Finder->getASTContext();
  const std::optional<Token> NextTok =
      utils::lexer::findNextTokenSkippingComments(InitRange.getBegin(),
                                                  Context.getSourceManager(),
                                                  Context.getLangOpts());
  if (!NextTok)
    return true;
  return NextTok->getLocation() != InitRange.getEnd();
}
} // namespace

void RedundantMemberInitCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreBaseInCopyConstructors",
                IgnoreBaseInCopyConstructors);
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void RedundantMemberInitCheck::registerMatchers(MatchFinder *Finder) {
  auto ConstructorMatcher =
      cxxConstructExpr(
          argumentCountIs(0),
          hasDeclaration(cxxConstructorDecl(
              ofClass(cxxRecordDecl(unless(isTriviallyDefaultConstructible()))
                          .bind("class")))),
          IgnoreMacros
              ? unless(initListContainsMacro())
              : static_cast<ast_matchers::internal::Matcher<CXXConstructExpr>>(
                    anything()))
          .bind("construct");

  auto HasUnionAsParent = hasParent(recordDecl(isUnion()));

  auto HasTypeEqualToConstructorClass = hasType(qualType(
      hasCanonicalType(qualType(hasDeclaration(equalsBoundNode("class"))))));

  Finder->addMatcher(
      cxxConstructorDecl(
          unless(isDelegatingConstructor()), ofClass(unless(isUnion())),
          forEachConstructorInitializer(
              cxxCtorInitializer(
                  withInitializer(ConstructorMatcher),
                  anyOf(isBaseInitializer(),
                        forField(fieldDecl(unless(hasType(isConstQualified())),
                                           unless(HasUnionAsParent),
                                           HasTypeEqualToConstructorClass))))
                  .bind("init")))
          .bind("constructor"),
      this);

  Finder->addMatcher(fieldDecl(hasInClassInitializer(ConstructorMatcher),
                               HasTypeEqualToConstructorClass,
                               unless(HasUnionAsParent))
                         .bind("field"),
                     this);
}

void RedundantMemberInitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Construct = Result.Nodes.getNodeAs<CXXConstructExpr>("construct");

  if (const auto *Field = Result.Nodes.getNodeAs<FieldDecl>("field")) {
    const Expr *Init = Field->getInClassInitializer();
    auto Diag =
        diag(Construct->getExprLoc(), "initializer for member %0 is redundant")
        << Field;
    if (!Init->getBeginLoc().isMacroID() && !Init->getEndLoc().isMacroID())
      Diag << FixItHint::CreateRemoval(getFullInitRangeInclWhitespaces(
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
