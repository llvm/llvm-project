//===----------------------------------------------------------------------===//
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
#include "llvm/ADT/SmallVectorWithFlags.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace clang::tidy::bugprone {

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
  bool getCheckFunctionCalls() const { return AssertMacros.getFlag(0); }
  const StringRef RawAssertList;
  llvm::SmallVectorWithFlags<StringRef, 5> AssertMacros;
  const std::vector<StringRef> IgnoredFunctions;

public:
  AssertSideEffectCheck(const AssertSideEffectCheck &) = default;
  AssertSideEffectCheck(AssertSideEffectCheck &&) = default;
  AssertSideEffectCheck &operator=(const AssertSideEffectCheck &) = delete;
  AssertSideEffectCheck &operator=(AssertSideEffectCheck &&) = delete;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_ASSERTSIDEEFFECTCHECK_H
