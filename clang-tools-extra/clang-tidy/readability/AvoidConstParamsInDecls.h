//===--- AvoidConstParamsInDecls.h - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AVOID_CONST_PARAMS_IN_DECLS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AVOID_CONST_PARAMS_IN_DECLS_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace readability {

// Detect function declarations that have const value parameters and discourage
// them.
class AvoidConstParamsInDecls : public ClangTidyCheck {
public:
  AvoidConstParamsInDecls(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AVOID_CONST_PARAMS_IN_DECLS_H
