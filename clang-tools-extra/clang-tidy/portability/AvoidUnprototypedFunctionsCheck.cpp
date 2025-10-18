//===--- AvoidUnprototypedFunctionsCheck.cpp - clang-tidy -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidUnprototypedFunctionsCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::portability {

void AvoidUnprototypedFunctionsCheck::registerMatchers(MatchFinder *Finder) {
  auto FunctionTypeMatcher =
      forEachDescendant(functionType(unless(functionProtoType())));
  Finder->addMatcher(declaratorDecl(FunctionTypeMatcher).bind("declaratorDecl"),
                     this);
  Finder->addMatcher(typedefDecl(FunctionTypeMatcher).bind("typedefDecl"),
                     this);
}

void AvoidUnprototypedFunctionsCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *MatchedTypedefDecl =
          Result.Nodes.getNodeAs<TypedefDecl>("typedefDecl")) {
    diag(MatchedTypedefDecl->getLocation(),
         "avoid unprototyped functions in typedef; explicitly add a 'void' "
         "parameter if the function takes no arguments");
    return;
  }

  const auto *MatchedDeclaratorDecl =
      Result.Nodes.getNodeAs<Decl>("declaratorDecl");
  if (!MatchedDeclaratorDecl) {
    return;
  }

  if (const auto *MatchedFunctionDecl =
          llvm::dyn_cast<FunctionDecl>(MatchedDeclaratorDecl)) {
    if (MatchedFunctionDecl->isMain() ||
        MatchedFunctionDecl->hasWrittenPrototype())
      return;

    diag(MatchedFunctionDecl->getLocation(),
         "avoid unprototyped function declarations; explicitly spell out a "
         "single 'void' parameter if the function takes no argument");
    return;
  }

  diag(MatchedDeclaratorDecl->getLocation(),
       "avoid unprototyped functions in type specifiers; explicitly add a "
       "'void' parameter if the function takes no arguments");
}

} // namespace clang::tidy::portability
