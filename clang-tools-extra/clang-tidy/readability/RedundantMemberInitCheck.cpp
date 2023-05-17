//===--- RedundantMemberInitCheck.cpp - clang-tidy-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

namespace clang::tidy::readability {

void RedundantMemberInitCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreBaseInCopyConstructors",
                IgnoreBaseInCopyConstructors);
}

void RedundantMemberInitCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxConstructorDecl(
          unless(isDelegatingConstructor()), ofClass(unless(isUnion())),
          forEachConstructorInitializer(
              cxxCtorInitializer(
                  withInitializer(
                      cxxConstructExpr(
                          hasDeclaration(
                              cxxConstructorDecl(ofClass(cxxRecordDecl(
                                  unless(isTriviallyDefaultConstructible()))))))
                          .bind("construct")),
                  unless(forField(hasType(isConstQualified()))),
                  unless(forField(hasParent(recordDecl(isUnion())))))
                  .bind("init")))
          .bind("constructor"),
      this);
}

void RedundantMemberInitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Init = Result.Nodes.getNodeAs<CXXCtorInitializer>("init");
  const auto *Construct = Result.Nodes.getNodeAs<CXXConstructExpr>("construct");
  const auto *ConstructorDecl =
      Result.Nodes.getNodeAs<CXXConstructorDecl>("constructor");

  if (IgnoreBaseInCopyConstructors && ConstructorDecl->isCopyConstructor() &&
      Init->isBaseInitializer())
    return;

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

} // namespace clang::tidy::readability
