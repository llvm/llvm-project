//===--- ComparisonInTempFailureRetryCheck.h - clang-tidy--------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_COMPARISONINTEMPFAILURERETRYCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_COMPARISONINTEMPFAILURERETRYCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace android {

/// Attempts to catch calls to TEMP_FAILURE_RETRY with a top-level comparison
/// operation, like `TEMP_FAILURE_RETRY(read(...) != N)`. In these cases, the
/// comparison should go outside of the TEMP_FAILURE_RETRY.
///
/// TEMP_FAILURE_RETRY is a macro provided by both glibc and Bionic.
class ComparisonInTempFailureRetryCheck : public ClangTidyCheck {
public:
  ComparisonInTempFailureRetryCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace android
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_COMPARISONINTEMPFAILURERETRYCHECK_H
