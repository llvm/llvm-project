//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SPURIOUSLYWAKEUPFUNCTIONSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SPURIOUSLYWAKEUPFUNCTIONSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Finds ``cnd_wait``, ``cnd_timedwait``, ``wait``, ``wait_for``, or
/// ``wait_until`` function calls when the function is not invoked from a loop
/// that checks whether a condition predicate holds or the function has a
/// condition parameter.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/spuriously-wake-up-functions.html
class SpuriouslyWakeUpFunctionsCheck : public ClangTidyCheck {
public:
  SpuriouslyWakeUpFunctionsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SPURIOUSLYWAKEUPFUNCTIONSCHECK_H
