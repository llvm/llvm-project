//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_SIGNED_BITWISE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_SIGNED_BITWISE_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::hicpp {

/// This check implements the rule 5.6.1 of the HICPP Standard, which disallows
/// bitwise operations on signed integer types.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/hicpp/signed-bitwise.html
class SignedBitwiseCheck : public ClangTidyCheck {
public:
  SignedBitwiseCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  bool IgnorePositiveIntegerLiterals;
};

} // namespace clang::tidy::hicpp

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_SIGNED_BITWISE_H
