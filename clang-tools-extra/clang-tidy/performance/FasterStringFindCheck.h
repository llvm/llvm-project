//===--- FasterStringFindCheck.h - clang-tidy--------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_FASTER_STRING_FIND_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_FASTER_STRING_FIND_H

#include "../ClangTidy.h"

#include <string>
#include <vector>

namespace clang {
namespace tidy {
namespace performance {

/// Optimize calls to std::string::find() and friends when the needle passed is
/// a single character string literal.
/// The character literal overload is more efficient.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance-faster-string-find.html
class FasterStringFindCheck : public ClangTidyCheck {
public:
  FasterStringFindCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const std::vector<std::string> StringLikeClasses;
};

} // namespace performance
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_FASTER_STRING_FIND_H
