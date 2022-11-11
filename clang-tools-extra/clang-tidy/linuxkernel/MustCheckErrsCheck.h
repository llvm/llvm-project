//===--- MustCheckErrsCheck.h - clang-tidy ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LINUXKERNEL_MUSTCHECKERRSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LINUXKERNEL_MUSTCHECKERRSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace linuxkernel {

/// Checks Linux kernel code to see if it uses the results from the functions in
/// linux/err.h. Also checks to see if code uses the results from functions that
/// directly return a value from one of these error functions.
///
/// This is important in the Linux kernel because ERR_PTR, PTR_ERR, IS_ERR,
/// IS_ERR_OR_NULL, ERR_CAST, and PTR_ERR_OR_ZERO return values must be checked,
/// since positive pointers and negative error codes are being used in the same
/// context. These functions are marked with
/// __attribute__((warn_unused_result)), but some kernel versions do not have
/// this warning enabled for clang.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/linuxkernel/must-use-errs.html
class MustCheckErrsCheck : public ClangTidyCheck {
public:
  MustCheckErrsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace linuxkernel
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LINUXKERNEL_MUSTCHECKERRSCHECK_H
