//===--- AssertSideEffectCheck.h - clang-tidy -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_ASSERTSIDEEFFECTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_ASSERTSIDEEFFECTCHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace clang {
namespace tidy {
namespace bugprone {

/// Finds `assert()` with side effect.
///
/// The condition of `assert()` is evaluated only in debug builds so a
/// condition with side effect can cause different behavior in debug / release
/// builds.
///
/// There are two options:
///
///   - `AssertMacros`: A comma-separated list of the names of assert macros to
///     be checked.
///   - `CheckFunctionCalls`: Whether to treat non-const member and non-member
///     functions as they produce side effects. Disabled by default because it
///     can increase the number of false positive warnings.
class AssertSideEffectCheck : public ClangTidyCheck {
public:
  AssertSideEffectCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool CheckFunctionCalls;
  const std::string RawAssertList;
  SmallVector<StringRef, 5> AssertMacros;
  const std::vector<std::string> IgnoredFunctions;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_ASSERTSIDEEFFECTCHECK_H
