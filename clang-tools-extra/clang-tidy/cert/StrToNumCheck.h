//===--- StrToNumCheck.h - clang-tidy----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_STRTONUMCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_STRTONUMCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::cert {

/// Guards against use of string conversion functions that do not have
/// reasonable error handling for conversion errors.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cert/err34-c.html
class StrToNumCheck : public ClangTidyCheck {
public:
  StrToNumCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::cert

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_STRTONUMCHECK_H
