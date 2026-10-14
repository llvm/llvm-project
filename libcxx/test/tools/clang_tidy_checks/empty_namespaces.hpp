//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-tidy/ClangTidyCheck.h"

namespace libcpp {
class empty_namespaces : public clang::tidy::ClangTidyCheck {
public:
  empty_namespaces(llvm::StringRef, clang::tidy::ClangTidyContext*);
  void registerMatchers(clang::ast_matchers::MatchFinder*) override;
  void check(const clang::ast_matchers::MatchFinder::MatchResult&) override;
};
} // namespace libcpp
