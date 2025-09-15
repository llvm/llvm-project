//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_DARWIN_AVOIDSPINLOCKCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_DARWIN_AVOIDSPINLOCKCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::darwin {

/// Finds usages of OSSpinlock, which is deprecated due to potential livelock
/// problems.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/darwin/avoid-spinlock.html
class AvoidSpinlockCheck : public ClangTidyCheck {
public:
  AvoidSpinlockCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::darwin

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_DARWIN_AVOIDSPINLOCKCHECK_H
