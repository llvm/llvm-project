//===--- DurationDivisionCheck.h - clang-tidy--------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_DURATIONDIVISIONCHECK_H_
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_DURATIONDIVISIONCHECK_H_

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace abseil {

// Find potential incorrect uses of integer division of absl::Duration objects.
//
// For the user-facing documentation see: 
// http://clang.llvm.org/extra/clang-tidy/checks/abseil-duration-division.html

class DurationDivisionCheck : public ClangTidyCheck {
public:
  using ClangTidyCheck::ClangTidyCheck;
  void registerMatchers(ast_matchers::MatchFinder *finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &result) override;
};

} // namespace abseil
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_DURATIONDIVISIONCHECK_H_
