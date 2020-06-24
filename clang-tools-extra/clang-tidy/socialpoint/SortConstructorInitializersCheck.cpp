//===--- SortConstructorInitializersCheck.cpp - clang-tidy ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SortConstructorInitializersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace socialpoint {

void SortConstructorInitializersCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(cxxConstructorDecl(unless(isExpansionInSystemHeader()))
                         .bind("constructor"),
                     this);
}

void SortConstructorInitializersCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (auto *Decl = Result.Nodes.getNodeAs<CXXConstructorDecl>("constructor")) {
    process(Decl, *Result.SourceManager, Result.Context->getLangOpts());
  }
}

void SortConstructorInitializersCheck::process(const CXXConstructorDecl *Decl,
                                               SourceManager &SourceMgr,
                                               const LangOptions &LangOpts) {
  if (Decl->getNumCtorInitializers()) {
    bool AlreadySorted = true;
    std::string fixStorage;
    llvm::raw_string_ostream FixToApply(fixStorage);

    SourceRange RangeToFix;
    bool FirstInitializer = true;

    for (const CXXCtorInitializer *Init : Decl->inits()) {
      if (!Init->isMemberInitializer() || !Init->isWritten()) {
        continue;
      }
      SourceRange InitRange = Init->getSourceRange();
      if (FirstInitializer) {
        RangeToFix = InitRange;
      } else {
        if (InitRange.getBegin() < RangeToFix.getBegin()) {
          RangeToFix.setBegin(InitRange.getBegin());
          AlreadySorted = false;
        }
        if (InitRange.getEnd() > RangeToFix.getEnd()) {
          RangeToFix.setEnd(InitRange.getEnd());
        }
        FixToApply << ",";
      }
      FixToApply << Lexer::getSourceText(
          CharSourceRange::getTokenRange(InitRange), SourceMgr, LangOpts);
      FirstInitializer = false;
    }

    if (!AlreadySorted) {
      auto CharRangeToFix = CharSourceRange::getTokenRange(RangeToFix);
      diag(CharRangeToFix.getBegin(), "Member initializers are not sorted: %0")
          << Lexer::getSourceText(CharRangeToFix, SourceMgr, LangOpts)
          << FixItHint::CreateReplacement(CharRangeToFix, FixToApply.str());
    }
  }
}

} // namespace socialpoint
} // namespace tidy
} // namespace clang
