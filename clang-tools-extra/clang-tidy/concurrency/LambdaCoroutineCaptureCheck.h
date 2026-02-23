//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CONCURRENCY_LAMBDACOROUTINECAPTURECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CONCURRENCY_LAMBDACOROUTINECAPTURECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::concurrency {

/// Finds lambda coroutines that capture variables without using the C++23
/// "deducing this" syntax, which can lead to use-after-free bugs.
///
/// When a lambda coroutine is passed to APIs that store the lambda temporarily
/// (e.g. ``seastar::future::then()``), the lambda object may be destroyed while
/// the coroutine is still suspended. Captures reference the lambda's storage,
/// so accessing them after the lambda is destroyed causes undefined behavior.
/// The C++23 explicit object parameter (``this auto``) moves captures into the
/// coroutine frame, decoupling their lifetime from the lambda object.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/concurrency/lambda-coroutine-capture.html
class LambdaCoroutineCaptureCheck : public ClangTidyCheck {
public:
  LambdaCoroutineCaptureCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus23;
  }
};

} // namespace clang::tidy::concurrency

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CONCURRENCY_LAMBDACOROUTINECAPTURECHECK_H
