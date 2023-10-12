//===--- UndefinedMemoryManipulationCheck.h - clang-tidy---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNDEFINED_MEMORY_MANIPULATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNDEFINED_MEMORY_MANIPULATION_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Finds calls of memory manipulation functions ``memset()``, ``memcpy()`` and
/// ``memmove()`` on non-TriviallyCopyable objects resulting in undefined
/// behavior.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/undefined-memory-manipulation.html
class UndefinedMemoryManipulationCheck : public ClangTidyCheck {
public:
  UndefinedMemoryManipulationCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNDEFINED_MEMORY_MANIPULATION_H
