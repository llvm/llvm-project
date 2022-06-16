//===--- PreferRegisterOverUnsignedCheck.h - clang-tidy ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_PREFERREGISTEROVERUNSIGNEDCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_PREFERREGISTEROVERUNSIGNEDCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace llvm_check {

/// Historically, LLVM has used `unsigned` to represent registers. Since then
/// a `Register` object has been introduced for improved type-safety and make
/// the code more explicit.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/llvm/prefer-register-over-unsigned.html
class PreferRegisterOverUnsignedCheck : public ClangTidyCheck {
public:
  PreferRegisterOverUnsignedCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace llvm_check
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_PREFERREGISTEROVERUNSIGNEDCHECK_H
