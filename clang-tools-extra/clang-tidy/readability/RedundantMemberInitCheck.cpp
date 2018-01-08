//===--- RedundantMemberInitCheck.cpp - clang-tidy-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RedundantMemberInitCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include <algorithm>

using namespace clang::ast_matchers;
using namespace clang::tidy::matchers;

namespace clang {
namespace tidy {
namespace readability {

void RedundantMemberInitCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  auto Construct =
      cxxConstructExpr(
          hasDeclaration(cxxConstructorDecl(hasParent(
              cxxRecordDecl(unless(isTriviallyDefaultConstructible()))))))
          .bind("construct");

  Finder->addMatcher(
      cxxConstructorDecl(
          unless(isDelegatingConstructor()),
          ofClass(unless(
              anyOf(isUnion(), ast_matchers::isTemplateInstantiation()))),
          forEachConstructorInitializer(
              cxxCtorInitializer(isWritten(),
                                 withInitializer(ignoringImplicit(Construct)),
                                 unless(forField(hasType(isConstQualified()))),
                                 unless(forField(hasParent(recordDecl(isUnion())))))
                  .bind("init"))),
      this);
}

void RedundantMemberInitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Init = Result.Nodes.getNodeAs<CXXCtorInitializer>("init");
  const auto *Construct = Result.Nodes.getNodeAs<CXXConstructExpr>("construct");

  if (Construct->getNumArgs() == 0 ||
      Construct->getArg(0)->isDefaultArgument()) {
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
}

} // namespace readability
} // namespace tidy
} // namespace clang
