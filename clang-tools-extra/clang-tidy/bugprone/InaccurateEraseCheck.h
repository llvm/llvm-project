//===--- InaccurateEraseCheck.h - clang-tidy---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INACCURATEERASECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INACCURATEERASECHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Checks for inaccurate use of the `erase()` method.
///
/// Algorithms like `remove()` do not actually remove any element from the
/// container but return an iterator to the first redundant element at the end
/// of the container. These redundant elements must be removed using the
/// `erase()` method. This check warns when not all of the elements will be
/// removed due to using an inappropriate overload.
class InaccurateEraseCheck : public ClangTidyCheck {
public:
  InaccurateEraseCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INACCURATEERASECHECK_H
