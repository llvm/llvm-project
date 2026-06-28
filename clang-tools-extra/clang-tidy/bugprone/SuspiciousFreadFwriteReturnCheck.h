//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSFREADFWRITERETURNCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSFREADFWRITERETURNCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Finds suspicious checks of the return value of `fread` and `fwrite`.
///
/// Developers sometimes mistakenly treat the result like the `ssize_t`
/// return value of POSIX `read` and `write`. Unlike those functions,
/// `fread` and `fwrite` return the number of elements transferred as a
/// `size_t`. When more than one element is requested, comparing the result
/// against zero does not detect partial reads or writes. Correct code should
/// compare the returned element count against the requested `nmemb`.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/suspicious-fread-fwrite-return.html
class SuspiciousFreadFwriteReturnCheck : public ClangTidyCheck {
public:
  SuspiciousFreadFwriteReturnCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSFREADFWRITERETURNCHECK_H
