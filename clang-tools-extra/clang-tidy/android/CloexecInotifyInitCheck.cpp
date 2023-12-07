//===--- CloexecInotifyInitCheck.cpp - clang-tidy--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CloexecInotifyInitCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::android {

void CloexecInotifyInitCheck::registerMatchers(MatchFinder *Finder) {
  registerMatchersImpl(
      Finder, functionDecl(returns(isInteger()), hasName("inotify_init")));
}

void CloexecInotifyInitCheck::check(const MatchFinder::MatchResult &Result) {
  replaceFunc(Result, /*WarningMsg=*/
              "prefer inotify_init() to inotify_init1() "
              "because inotify_init1() allows IN_CLOEXEC",
              /*FixMsg=*/"inotify_init1(IN_CLOEXEC)");
}

} // namespace clang::tidy::android
