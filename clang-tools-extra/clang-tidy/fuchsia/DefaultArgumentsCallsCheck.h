//===--- DefaultArgumentsCallsCheck.h - clang-tidy --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_DEFAULT_ARGUMENTS_CALLS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_DEFAULT_ARGUMENTS_CALLS_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace fuchsia {

/// Default arguments are not allowed in called functions.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/fuchsia-default-arguments-calls.html
class DefaultArgumentsCallsCheck : public ClangTidyCheck {
public:
  DefaultArgumentsCallsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace fuchsia
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_DEFAULT_ARGUMENTS_CALLS_H
