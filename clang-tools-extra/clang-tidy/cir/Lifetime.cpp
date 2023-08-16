//===--- Lifetime.cpp - clang-tidy ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Lifetime.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cir {

void Lifetime::registerMatchers(MatchFinder *Finder) {
  //   Finder->addMatcher(callExpr().bind("CE"), this);
  // assert(0 && "BOOM0!");
}

void Lifetime::check(const MatchFinder::MatchResult &Result) {
  // assert(0 && "BOOM1!");
}

void Lifetime::onEndOfTranslationUnit() { assert(0 && "BOOM2!"); }
} // namespace clang::tidy::cir
