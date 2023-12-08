//===--- MissingHashCheck.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MissingHashCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::objc {

namespace {

AST_MATCHER_P(ObjCImplementationDecl, hasInterface,
              ast_matchers::internal::Matcher<ObjCInterfaceDecl>, Base) {
  const ObjCInterfaceDecl *InterfaceDecl = Node.getClassInterface();
  return Base.matches(*InterfaceDecl, Finder, Builder);
}

AST_MATCHER_P(ObjCContainerDecl, hasInstanceMethod,
              ast_matchers::internal::Matcher<ObjCMethodDecl>, Base) {
  // Check each instance method against the provided matcher.
  for (const auto *I : Node.instance_methods()) {
    if (Base.matches(*I, Finder, Builder))
      return true;
  }
  return false;
}

} // namespace

void MissingHashCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      objcMethodDecl(
          hasName("isEqual:"), isInstanceMethod(),
          hasDeclContext(objcImplementationDecl(
                             hasInterface(isDirectlyDerivedFrom("NSObject")),
                             unless(hasInstanceMethod(hasName("hash"))))
                             .bind("impl"))),
      this);
}

void MissingHashCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ID = Result.Nodes.getNodeAs<ObjCImplementationDecl>("impl");
  diag(ID->getLocation(), "%0 implements -isEqual: without implementing -hash")
      << ID;
}

} // namespace clang::tidy::objc
