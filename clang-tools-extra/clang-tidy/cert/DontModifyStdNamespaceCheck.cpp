//===--- DontModifyStdNamespaceCheck.cpp - clang-tidy----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DontModifyStdNamespaceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cert {

void DontModifyStdNamespaceCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  Finder->addMatcher(
      namespaceDecl(unless(isExpansionInSystemHeader()),
                    anyOf(hasName("std"), hasName("posix")),
                    has(decl(unless(anyOf(
                        functionDecl(isExplicitTemplateSpecialization()),
                        cxxRecordDecl(isExplicitTemplateSpecialization()))))))
          .bind("nmspc"),
      this);
}

void DontModifyStdNamespaceCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *N = Result.Nodes.getNodeAs<NamespaceDecl>("nmspc");

  // Only consider top level namespaces.
  if (N->getParent() != Result.Context->getTranslationUnitDecl())
    return;

  diag(N->getLocation(),
       "modification of %0 namespace can result in undefined behavior")
      << N;
}

} // namespace cert
} // namespace tidy
} // namespace clang
