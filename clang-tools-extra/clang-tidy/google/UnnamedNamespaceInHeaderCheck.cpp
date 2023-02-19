//===--- UnnamedNamespaceInHeaderCheck.cpp - clang-tidy ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnnamedNamespaceInHeaderCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::google::build {

UnnamedNamespaceInHeaderCheck::UnnamedNamespaceInHeaderCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {
  std::optional<StringRef> HeaderFileExtensionsOption =
      Options.get("HeaderFileExtensions");
  RawStringHeaderFileExtensions =
      HeaderFileExtensionsOption.value_or(utils::defaultHeaderFileExtensions());
  if (HeaderFileExtensionsOption) {
    if (!utils::parseFileExtensions(RawStringHeaderFileExtensions,
                                    HeaderFileExtensions,
                                    utils::defaultFileExtensionDelimiters())) {
      this->configurationDiag("Invalid header file extension: '%0'")
          << RawStringHeaderFileExtensions;
    }
  } else
    HeaderFileExtensions = Context->getHeaderFileExtensions();
}

void UnnamedNamespaceInHeaderCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "HeaderFileExtensions", RawStringHeaderFileExtensions);
}

void UnnamedNamespaceInHeaderCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
    Finder->addMatcher(namespaceDecl(isAnonymous()).bind("anonymousNamespace"),
                       this);
}

void UnnamedNamespaceInHeaderCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *N = Result.Nodes.getNodeAs<NamespaceDecl>("anonymousNamespace");
  SourceLocation Loc = N->getBeginLoc();
  if (!Loc.isValid())
    return;

  if (utils::isPresumedLocInHeaderFile(Loc, *Result.SourceManager,
                                       HeaderFileExtensions))
    diag(Loc, "do not use unnamed namespaces in header files");
}

} // namespace clang::tidy::google::build
