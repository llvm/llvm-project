//===--- NoAssemblerCheck.h - clang-tidy-------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_NO_ASSEMBLER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_NO_ASSEMBLER_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::hicpp {

/// Find assembler statements. No fix is offered.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/hicpp/no-assembler.html
class NoAssemblerCheck : public ClangTidyCheck {
public:
  NoAssemblerCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::hicpp

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_NO_ASSEMBLER_H
