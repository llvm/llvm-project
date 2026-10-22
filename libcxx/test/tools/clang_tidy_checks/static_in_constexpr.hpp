//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_TEST_TOOLS_CLANG_TIDY_CHECKS_STATIC_IN_CONSTEXPR_HPP
#define LIBCXX_TEST_TOOLS_CLANG_TIDY_CHECKS_STATIC_IN_CONSTEXPR_HPP

#include "clang-tidy/ClangTidyCheck.h"

namespace libcpp {

class static_in_constexpr : public clang::tidy::ClangTidyCheck {
public:
  static_in_constexpr(llvm::StringRef name, clang::tidy::ClangTidyContext* context)
      : clang::tidy::ClangTidyCheck(name, context) {}
  void registerMatchers(clang::ast_matchers::MatchFinder* finder) override;
  void check(const clang::ast_matchers::MatchFinder::MatchResult& result) override;

  bool isLanguageVersionSupported(const clang::LangOptions& lang_opts) const override { return !lang_opts.CPlusPlus23; }
};

} // namespace libcpp

#endif // LIBCXX_TEST_TOOLS_CLANG_TIDY_CHECKS_STATIC_IN_CONSTEXPR_HPP
