//===--- MtUnsafeCheck.h - clang-tidy ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CONCURRENCY_MTUNSAFECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CONCURRENCY_MTUNSAFECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::concurrency {

/// Checks that non-thread-safe functions are not used.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/concurrency/mt-unsafe.html
class MtUnsafeCheck : public ClangTidyCheck {
public:
  MtUnsafeCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  enum class FunctionSet {
    Posix,
    Glibc,
    Any,
  };

private:
  const FunctionSet FuncSet;
};

} // namespace clang::tidy::concurrency

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CONCURRENCY_MTUNSAFECHECK_H
