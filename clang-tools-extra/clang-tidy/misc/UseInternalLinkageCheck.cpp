//===--- UseInternalLinkageCheck.cpp - clang-tidy--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseInternalLinkageCheck.h"
#include "../utils/FileExtensionsUtils.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

namespace {

AST_MATCHER(Decl, isFirstDecl) { return Node.isFirstDecl(); }

static bool isInMainFile(SourceLocation L, SourceManager &SM,
                         const FileExtensionsSet &HeaderFileExtensions) {
  for (;;) {
    if (utils::isSpellingLocInHeaderFile(L, SM, HeaderFileExtensions))
      return false;
    if (SM.isInMainFile(L))
      return true;
    // not in header file but not in main file
    L = SM.getIncludeLoc(SM.getFileID(L));
    if (L.isValid())
      continue;
    // Conservative about the unknown
    return false;
  }
}

AST_MATCHER_P(Decl, isAllRedeclsInMainFile, FileExtensionsSet,
              HeaderFileExtensions) {
  return llvm::all_of(Node.redecls(), [&](const Decl *D) {
    return isInMainFile(D->getLocation(),
                        Finder->getASTContext().getSourceManager(),
                        HeaderFileExtensions);
  });
}

AST_POLYMORPHIC_MATCHER(isExternStorageClass,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                        VarDecl)) {
  return Node.getStorageClass() == SC_Extern;
}

} // namespace

void UseInternalLinkageCheck::registerMatchers(MatchFinder *Finder) {
  auto Common =
      allOf(isFirstDecl(), isAllRedeclsInMainFile(HeaderFileExtensions),
            unless(anyOf(
                // 1. internal linkage
                isStaticStorageClass(), isInAnonymousNamespace(),
                // 2. explicit external linkage
                isExternStorageClass(), isExternC(),
                // 3. template
                isExplicitTemplateSpecialization(),
                // 4. friend
                hasAncestor(friendDecl()))));
  Finder->addMatcher(
      functionDecl(Common, unless(cxxMethodDecl()), unless(isMain()))
          .bind("fn"),
      this);
  Finder->addMatcher(varDecl(Common, hasGlobalStorage()).bind("var"), this);
}

static constexpr StringRef Message =
    "%0 %1 can be made static or moved into an anonymous namespace "
    "to enforce internal linkage";

void UseInternalLinkageCheck::check(const MatchFinder::MatchResult &Result) {
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

} // namespace clang::tidy::misc
