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

namespace clang {
namespace tidy {
namespace misc {
namespace {
AST_POLYMORPHIC_MATCHER(isStatic, AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                                  VarDecl)) {
  return Node.getStorageClass() == SC_Static;
}

AST_MATCHER(FunctionDecl, isMemberFunction) {
  return llvm::isa<CXXMethodDecl>(&Node);
}
AST_MATCHER(VarDecl, isStaticDataMember) { return Node.isStaticDataMember(); }
} // namespace

static bool isInAnonymousNamespace(const Decl *Decl) {
  const DeclContext *DC = Decl->getDeclContext();
  if (DC && DC->isNamespace()) {
    const auto *ND = llvm::cast<NamespaceDecl>(DC);
    if (ND && ND->isAnonymousNamespace())
      return true;
  }
  return false;
}

template <typename T>
void UseAnonymousNamespaceCheck::processMatch(const T *MatchedDecl) {
  StringRef Type = llvm::isa<VarDecl>(MatchedDecl) ? "variable" : "function";
  if (isInAnonymousNamespace(MatchedDecl))
    diag(MatchedDecl->getLocation(), "%0 %1 declared 'static' in "
                                     "anonymous namespace, remove 'static'")
        << Type << MatchedDecl;
  else
    diag(MatchedDecl->getLocation(),
         "%0 %1 declared 'static', move to anonymous namespace instead")
        << Type << MatchedDecl;
}

void UseAnonymousNamespaceCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(isStatic(), unless(isMemberFunction())).bind("func"), this);
  Finder->addMatcher(
      varDecl(isStatic(), unless(anyOf(isStaticLocal(), isStaticDataMember())))
          .bind("var"),
      this);
}

void UseAnonymousNamespaceCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("func"))
    processMatch(MatchedDecl);

  if (const auto *MatchedDecl = Result.Nodes.getNodeAs<VarDecl>("var"))
    processMatch(MatchedDecl);
}

} // namespace misc
} // namespace tidy
} // namespace clang
