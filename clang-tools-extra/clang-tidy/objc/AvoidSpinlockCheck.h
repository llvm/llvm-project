//===--- AvoidSpinlockCheck.h - clang-tidy-----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_AVOID_SPINLOCK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_AVOID_SPINLOCK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace objc {

/// Finds usages of OSSpinlock, which is deprecated due to potential livelock
/// problems.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/objc-avoid-spinlock.html
class AvoidSpinlockCheck : public ClangTidyCheck {
 public:
  AvoidSpinlockCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

}  // namespace objc
}  // namespace tidy
}  // namespace clang

#endif  // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_AVOID_SPINLOCK_H
