//===--- ExplicitConstructorCheck.h - clang-tidy ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_EXPLICITCONSTRUCTORCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_EXPLICITCONSTRUCTORCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace google {

/// Checks that all single-argument constructors are explicit.
///
/// See https://google.github.io/styleguide/cppguide.html#Explicit_Constructors
class ExplicitConstructorCheck : public ClangTidyCheck {
public:
  ExplicitConstructorCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace google
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_EXPLICITCONSTRUCTORCHECK_H
