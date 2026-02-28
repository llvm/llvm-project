//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_POINTERTOREFCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_POINTERTOREFCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Finds function parameters that are pointers but are always dereferenced
/// without null checks, suggesting they should be references instead.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/readability/pointer-to-ref.html
class PointerToRefCheck : public ClangTidyCheck {
public:
  PointerToRefCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_POINTERTOREFCHECK_H
