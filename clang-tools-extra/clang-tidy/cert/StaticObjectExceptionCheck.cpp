//===--- StaticObjectExceptionCheck.cpp - clang-tidy-----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "StaticObjectExceptionCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cert {

void StaticObjectExceptionCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  // Match any static or thread_local variable declaration that is initialized
  // with a constructor that can throw.
  Finder->addMatcher(
      varDecl(anyOf(hasThreadStorageDuration(), hasStaticStorageDuration()),
              hasInitializer(cxxConstructExpr(hasDeclaration(
                  cxxConstructorDecl(unless(isNoThrow()))
                      .bind("ctor")))))
          .bind("var"),
      this);
}

void StaticObjectExceptionCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *VD = Result.Nodes.getNodeAs<VarDecl>("var");
  const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructorDecl>("ctor");

  diag(VD->getLocation(),
       "construction of %0 with %select{static|thread_local}1 storage "
       "duration may throw an exception that cannot be caught")
      << VD << (VD->getStorageDuration() == SD_Static ? 0 : 1);
  diag(Ctor->getLocation(), "possibly throwing constructor declared here",
       DiagnosticIDs::Note);
}

} // namespace cert
} // namespace tidy
} // namespace clang
