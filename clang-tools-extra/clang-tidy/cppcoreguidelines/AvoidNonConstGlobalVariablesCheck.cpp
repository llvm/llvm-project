//===--- AvoidNonConstGlobalVariablesCheck.cpp - clang-tidy ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidNonConstGlobalVariablesCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

void AvoidNonConstGlobalVariablesCheck::registerMatchers(MatchFinder *Finder) {
  auto GlobalContext =
      varDecl(hasGlobalStorage(),
              hasDeclContext(anyOf(namespaceDecl(), translationUnitDecl())));

  auto GlobalVariable = varDecl(
      GlobalContext,
      unless(anyOf(
          isConstexpr(), hasType(isConstQualified()),
          hasType(referenceType())))); // References can't be changed, only the
                                       // data they reference can be changed.

  auto GlobalReferenceToNonConst =
      varDecl(GlobalContext, hasType(referenceType()),
              unless(hasType(references(qualType(isConstQualified())))));

  auto GlobalPointerToNonConst = varDecl(
      GlobalContext, hasType(pointerType(pointee(unless(isConstQualified())))));

  Finder->addMatcher(GlobalVariable.bind("non-const_variable"), this);
  Finder->addMatcher(GlobalReferenceToNonConst.bind("indirection_to_non-const"),
                     this);
  Finder->addMatcher(GlobalPointerToNonConst.bind("indirection_to_non-const"),
                     this);
}

void AvoidNonConstGlobalVariablesCheck::check(
    const MatchFinder::MatchResult &Result) {

  if (const auto *Variable =
          Result.Nodes.getNodeAs<VarDecl>("non-const_variable")) {
    diag(Variable->getLocation(), "variable %0 is non-const and globally "
                                  "accessible, consider making it const")
        << Variable; // FIXME: Add fix-it hint to Variable
    // Don't return early, a non-const variable may also be a pointer or
    // reference to non-const data.
  }

  if (const auto *VD =
          Result.Nodes.getNodeAs<VarDecl>("indirection_to_non-const")) {
    diag(VD->getLocation(),
         "variable %0 provides global access to a non-const object; consider "
         "making the %select{referenced|pointed-to}1 data 'const'")
        << VD
        << VD->getType()->isPointerType(); // FIXME: Add fix-it hint to Variable
  }
}

} // namespace clang::tidy::cppcoreguidelines
