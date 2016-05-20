//===--- AssertSideEffectCheck.h - clang-tidy -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_ASSERTSIDEEFFECTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_ASSERTSIDEEFFECTCHECK_H

#include "../ClangTidy.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace clang {
namespace tidy {
namespace misc {

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
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_ASSERTSIDEEFFECTCHECK_H
