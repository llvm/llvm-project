//===--- Lifetime.h - clang-tidy --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CIR_LIFETIME_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CIR_LIFETIME_H

#include "../ClangTidyCheck.h"
#include <optional>

namespace clang::tidy::cir {

class Lifetime : public ClangTidyCheck {
public:
  Lifetime(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;
};

} // namespace clang::tidy::cir

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CIR_LIFETIME_H
