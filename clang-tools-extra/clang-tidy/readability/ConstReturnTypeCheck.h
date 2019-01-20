//===--- ConstReturnTypeCheck.h - clang-tidy --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONSTRETURNTYPECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONSTRETURNTYPECHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// For any function whose return type is const-qualified, suggests removal of
/// the `const` qualifier from that return type.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-const-return-type.html
class ConstReturnTypeCheck : public ClangTidyCheck {
 public:
  using ClangTidyCheck::ClangTidyCheck;
  void registerMatchers(ast_matchers::MatchFinder* finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult& result) override;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONSTRETURNTYPECHECK_H
