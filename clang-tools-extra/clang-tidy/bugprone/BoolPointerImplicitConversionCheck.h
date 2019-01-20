//===--- BoolPointerImplicitConversionCheck.h - clang-tidy ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_BOOLPOINTERIMPLICITCONVERSIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_BOOLPOINTERIMPLICITCONVERSIONCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Checks for conditions based on implicit conversion from a bool pointer to
/// bool.
///
/// Example:
///
/// \code
///   bool *p;
///   if (p) {
///     // Never used in a pointer-specific way.
///   }
/// \endcode
class BoolPointerImplicitConversionCheck : public ClangTidyCheck {
public:
  BoolPointerImplicitConversionCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_BOOLPOINTERIMPLICITCONVERSIONCHECK_H
