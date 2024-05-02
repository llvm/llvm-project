//===--- UnnecessaryExternalLinkageCheck.cpp - clang-tidy
//---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnnecessaryExternalLinkageCheck.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/Specifiers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

AST_POLYMORPHIC_MATCHER(isFirstDecl,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                        VarDecl)) {
  return Node.isFirstDecl();
}

AST_MATCHER(Decl, isInMainFile) {
  for (const Decl *D : Node.redecls())
    if (!Finder->getASTContext().getSourceManager().isInMainFile(
            D->getLocation()))
      return false;
  return true;
}

AST_POLYMORPHIC_MATCHER(isExternStorageClass,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                        VarDecl)) {
  return Node.getStorageClass() == SC_Extern;
}

} // namespace

void UnnecessaryExternalLinkageCheck::registerMatchers(MatchFinder *Finder) {
  auto Common = allOf(isFirstDecl(), isInMainFile(),
                      unless(anyOf(
                          // 1. internal linkage
                          isStaticStorageClass(), isInAnonymousNamespace(),
                          // 2. explicit external linkage
                          isExternStorageClass(), isExternC(),
                          // 3. template
                          isExplicitTemplateSpecialization(),
                          clang::ast_matchers::isTemplateInstantiation(),
                          // 4. friend
                          hasAncestor(friendDecl()))));
  Finder->addMatcher(
      functionDecl(Common, unless(cxxMethodDecl()), unless(isMain()))
          .bind("fn"),
      this);
  Finder->addMatcher(varDecl(Common, hasGlobalStorage()).bind("var"), this);
}

static constexpr StringRef Message =
    "%0 %1 can be internal linkage, "
    "marking as static or using anonymous namespace can avoid external "
    "linkage.";

void UnnecessaryExternalLinkageCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>("fn")) {
    diag(FD->getLocation(), Message) << "function" << FD;
    return;
  }
  if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>("var")) {
    diag(VD->getLocation(), Message) << "variable" << VD;
    return;
  }
  llvm_unreachable("");
}

} // namespace clang::tidy::readability
