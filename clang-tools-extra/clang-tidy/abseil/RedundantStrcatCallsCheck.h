//===--- RedundantStrcatCallsCheck.h - clang-tidy----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_REDUNDANTSTRCATCALLSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_REDUNDANTSTRCATCALLSCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace abseil {

/// Flags redundant calls to absl::StrCat when the result is being passed to 
///	another call of absl::StrCat/absl::StrAppend. Also suggests a fix to 
///	collapse the calls.
/// Example:
///   StrCat(1, StrCat(2, 3))  ==>  StrCat(1, 2, 3)
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/abseil-redundant-strcat-calls.html
class RedundantStrcatCallsCheck : public ClangTidyCheck {
public:
  RedundantStrcatCallsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace abseil
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_REDUNDANTSTRCATCALLSCHECK_H
