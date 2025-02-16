//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-tidy/ClangTidyCheck.h"

namespace libcpp {
class robust_against_operator_ampersand : public clang::tidy::ClangTidyCheck {
  // At the moment libc++ is phasing out development on C++03.
  // To avoid testing in C++03, this test is automatically disabled in C++03
  // mode. (Doing this from the tests is a lot harder.)
  // TODO Remove after dropping C++03 support.
  bool disabled_;

public:
  robust_against_operator_ampersand(llvm::StringRef, clang::tidy::ClangTidyContext*);
  void registerMatchers(clang::ast_matchers::MatchFinder*) override;
  void check(const clang::ast_matchers::MatchFinder::MatchResult&) override;
};
} // namespace libcpp
