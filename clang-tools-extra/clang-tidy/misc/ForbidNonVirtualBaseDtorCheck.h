//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_FORBIDNONVIRTUALBASEDTORCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_FORBIDNONVIRTUALBASEDTORCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::misc {

/// Warns when a class or struct publicly inherits from a base class or struct
/// whose destructor is neither virtual nor protected, and the derived class
/// adds data members
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/misc/forbid-non-virtual-base-dtor.html
class ForbidNonVirtualBaseDtorCheck : public ClangTidyCheck {
public:
  ForbidNonVirtualBaseDtorCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_FORBIDNONVIRTUALBASEDTORCHECK_H
