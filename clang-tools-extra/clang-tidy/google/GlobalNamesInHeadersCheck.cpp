//===--- GlobalNamesInHeadersCheck.cpp - clang-tidy -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobalNamesInHeadersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::google::readability {

GlobalNamesInHeadersCheck::GlobalNamesInHeadersCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      RawStringHeaderFileExtensions(Options.getLocalOrGlobal(
          "HeaderFileExtensions", utils::defaultHeaderFileExtensions())) {
  if (!utils::parseFileExtensions(RawStringHeaderFileExtensions,
                                  HeaderFileExtensions,
                                  utils::defaultFileExtensionDelimiters())) {
    this->configurationDiag("Invalid header file extension: '%0'")
        << RawStringHeaderFileExtensions;
  }
}

void GlobalNamesInHeadersCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "HeaderFileExtensions", RawStringHeaderFileExtensions);
}

void GlobalNamesInHeadersCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(decl(anyOf(usingDecl(), usingDirectiveDecl()),
                          hasDeclContext(translationUnitDecl()))
                         .bind("using_decl"),
                     this);
}

void GlobalNamesInHeadersCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *D = Result.Nodes.getNodeAs<Decl>("using_decl");
  // If it comes from a macro, we'll assume it is fine.
  if (D->getBeginLoc().isMacroID())
    return;

  // Ignore if it comes from the "main" file ...
  if (Result.SourceManager->isInMainFile(
          Result.SourceManager->getExpansionLoc(D->getBeginLoc()))) {
    // unless that file is a header.
    if (!utils::isSpellingLocInHeaderFile(
            D->getBeginLoc(), *Result.SourceManager, HeaderFileExtensions))
      return;
  }

  if (const auto *UsingDirective = dyn_cast<UsingDirectiveDecl>(D)) {
    if (UsingDirective->getNominatedNamespace()->isAnonymousNamespace()) {
      // Anonymous namespaces inject a using directive into the AST to import
      // the names into the containing namespace.
      // We should not have them in headers, but there is another warning for
      // that.
      return;
    }
  }

  diag(D->getBeginLoc(),
       "using declarations in the global namespace in headers are prohibited");
}

} // namespace clang::tidy::google::readability
