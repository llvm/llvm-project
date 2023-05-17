//===--- UseAnonymousNamespaceCheck.cpp - clang-tidy ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseAnonymousNamespaceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {
namespace {
AST_POLYMORPHIC_MATCHER_P(isInHeaderFile,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                          VarDecl),
                          FileExtensionsSet, HeaderFileExtensions) {
  return utils::isExpansionLocInHeaderFile(
      Node.getBeginLoc(), Finder->getASTContext().getSourceManager(),
      HeaderFileExtensions);
}

AST_MATCHER(FunctionDecl, isMemberFunction) {
  return llvm::isa<CXXMethodDecl>(&Node);
}
AST_MATCHER(VarDecl, isStaticDataMember) { return Node.isStaticDataMember(); }
} // namespace

UseAnonymousNamespaceCheck::UseAnonymousNamespaceCheck(
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

void UseAnonymousNamespaceCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "HeaderFileExtensions", RawStringHeaderFileExtensions);
}

void UseAnonymousNamespaceCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(isStaticStorageClass(),
                   unless(anyOf(isInHeaderFile(HeaderFileExtensions),
                                isInAnonymousNamespace(), isMemberFunction())))
          .bind("x"),
      this);
  Finder->addMatcher(
      varDecl(isStaticStorageClass(),
              unless(anyOf(isInHeaderFile(HeaderFileExtensions),
                           isInAnonymousNamespace(), isStaticLocal(),
                           isStaticDataMember(), hasType(isConstQualified()))))
          .bind("x"),
      this);
}

void UseAnonymousNamespaceCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *MatchedDecl = Result.Nodes.getNodeAs<NamedDecl>("x")) {
    StringRef Type = llvm::isa<VarDecl>(MatchedDecl) ? "variable" : "function";
    diag(MatchedDecl->getLocation(),
         "%0 %1 declared 'static', move to anonymous namespace instead")
        << Type << MatchedDecl;
  }
}

} // namespace clang::tidy::misc
