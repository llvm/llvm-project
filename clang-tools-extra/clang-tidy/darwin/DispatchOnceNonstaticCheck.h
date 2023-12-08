//===--- DispatchOnceNonstaticCheck.h - clang-tidy --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_DARWIN_DISPATCHONCENONSTATICCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_DARWIN_DISPATCHONCENONSTATICCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::darwin {

/// Finds variables of type dispatch_once_t that do not have static or global
/// storage duration, as required by the libdispatch documentation.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/darwin/dispatch-once-nonstatic.html
class DispatchOnceNonstaticCheck : public ClangTidyCheck {
public:
  DispatchOnceNonstaticCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::darwin

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_DARWIN_DISPATCHONCENONSTATICCHECK_H
