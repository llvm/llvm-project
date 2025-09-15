//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CONCURRENCY_THREADCANCELTYPEASYNCHRONOUSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CONCURRENCY_THREADCANCELTYPEASYNCHRONOUSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::concurrency {

/// Finds ``pthread_setcanceltype`` function calls where a thread's
/// cancellation type is set to asynchronous.
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/concurrency/thread-canceltype-asynchronous.html
class ThreadCanceltypeAsynchronousCheck : public ClangTidyCheck {
public:
  ThreadCanceltypeAsynchronousCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::concurrency

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CONCURRENCY_THREADCANCELTYPEASYNCHRONOUSCHECK_H
