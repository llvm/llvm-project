//===--- StringCompareCheck.h - clang-tidy-----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STRING_COMPARE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STRING_COMPARE_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// This check flags all calls compare when used to check for string
/// equality or inequality.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-string-compare.html
class StringCompareCheck : public ClangTidyCheck {
public:
  StringCompareCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STRING_COMPARE_H
