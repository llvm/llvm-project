//===--- CloexecInotifyInitCheck.cpp - clang-tidy--------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CloexecInotifyInitCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

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

} // namespace android
} // namespace tidy
} // namespace clang
