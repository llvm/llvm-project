//===--- TwineLocalCheck.h - clang-tidy -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_TWINE_LOCAL_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_TWINE_LOCAL_CHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace llvm {

/// Looks for local `Twine` variables which are prone to use after frees and
/// should be generally avoided.
class TwineLocalCheck : public ClangTidyCheck {
public:
  TwineLocalCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace llvm
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_TWINE_LOCAL_CHECK_H
