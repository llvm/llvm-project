//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantAccessSpecifiersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void RedundantAccessSpecifiersCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxRecordDecl(has(accessSpecDecl())).bind("redundant-access-specifiers"),
      this);
}

void RedundantAccessSpecifiersCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl =
      Result.Nodes.getNodeAs<CXXRecordDecl>("redundant-access-specifiers");

  const AccessSpecDecl *LastASDecl = nullptr;
  for (DeclContext::specific_decl_iterator<AccessSpecDecl>
           AS(MatchedDecl->decls_begin()),
       ASEnd(MatchedDecl->decls_end());
       AS != ASEnd; ++AS) {
    const AccessSpecDecl *ASDecl = *AS;

    // Ignore macro expansions.
    if (ASDecl->getLocation().isMacroID()) {
      LastASDecl = ASDecl;
      continue;
    }

    if (LastASDecl == nullptr) {
      // First declaration.
      LastASDecl = ASDecl;

      if (CheckFirstDeclaration) {
        AccessSpecifier DefaultSpecifier =
            MatchedDecl->isClass() ? AS_private : AS_public;
        if (ASDecl->getAccess() == DefaultSpecifier) {
          diag(ASDecl->getLocation(),
               "redundant access specifier has the same accessibility as the "
               "implicit access specifier")
              << FixItHint::CreateRemoval(ASDecl->getSourceRange());
        }
      }

      continue;
    }

    if (LastASDecl->getAccess() == ASDecl->getAccess()) {
      // Ignore macro expansions.
      if (LastASDecl->getLocation().isMacroID()) {
        LastASDecl = ASDecl;
        continue;
      }

      diag(ASDecl->getLocation(),
           "redundant access specifier has the same accessibility as the "
           "previous access specifier")
          << FixItHint::CreateRemoval(ASDecl->getSourceRange());
      diag(LastASDecl->getLocation(), "previously declared here",
           DiagnosticIDs::Note);
    } else {
      LastASDecl = ASDecl;
    }
  }
}

} // namespace clang::tidy::readability
