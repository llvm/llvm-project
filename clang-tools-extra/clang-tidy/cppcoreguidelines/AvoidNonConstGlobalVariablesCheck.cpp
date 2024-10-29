//===--- AvoidNonConstGlobalVariablesCheck.cpp - clang-tidy ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidNonConstGlobalVariablesCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

AvoidNonConstGlobalVariablesCheck::AvoidNonConstGlobalVariablesCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowInternalLinkage(Options.get("AllowInternalLinkage", false)) {}

void AvoidNonConstGlobalVariablesCheck::registerMatchers(MatchFinder *Finder) {
  auto NamespaceMatcher = AllowInternalLinkage
                              ? namespaceDecl(unless(isAnonymous()))
                              : namespaceDecl();
  auto GlobalContext =
      varDecl(hasGlobalStorage(),
              hasDeclContext(anyOf(NamespaceMatcher, translationUnitDecl())));

  auto GlobalVariable = varDecl(
      GlobalContext,
      AllowInternalLinkage ? varDecl(unless(isStaticStorageClass()))
                           : varDecl(),
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

void AvoidNonConstGlobalVariablesCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowInternalLinkage", AllowInternalLinkage);
}

} // namespace clang::tidy::cppcoreguidelines
