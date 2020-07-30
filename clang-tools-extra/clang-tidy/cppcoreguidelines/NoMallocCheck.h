//===--- NoMallocCheck.h - clang-tidy----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_NO_MALLOC_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_NO_MALLOC_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// This checker is concerned with C-style memory management and suggest modern
/// alternatives to it.
/// The check is only enabled in C++. For analyzing malloc calls see Clang
/// Static Analyzer - unix.Malloc.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-no-malloc.html
class NoMallocCheck : public ClangTidyCheck {
public:
  /// Construct Checker and read in configuration for function names.
  NoMallocCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        AllocList(Options.get("Allocations", "::malloc;::calloc")),
        ReallocList(Options.get("Reallocations", "::realloc")),
        DeallocList(Options.get("Deallocations", "::free")) {}

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

  /// Make configuration of checker discoverable.
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  /// Registering for malloc, calloc, realloc and free calls.
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;

  /// Checks matched function calls and gives suggestion to modernize the code.
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  /// Semicolon-separated list of fully qualified names of memory allocation
  /// functions the check warns about. Defaults to `::malloc;::calloc`.
  const std::string AllocList;
  /// Semicolon-separated list of fully qualified names of memory reallocation
  /// functions the check warns about. Defaults to `::realloc`.
  const std::string ReallocList;
  /// Semicolon-separated list of fully qualified names of memory deallocation
  /// functions the check warns about. Defaults to `::free`.
  const std::string DeallocList;
};

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_NO_MALLOC_H
