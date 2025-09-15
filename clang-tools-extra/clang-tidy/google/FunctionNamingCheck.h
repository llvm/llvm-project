//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OBJC_FUNCTION_NAMING_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OBJC_FUNCTION_NAMING_CHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/StringRef.h"

namespace clang::tidy::google::objc {

/// Finds function names that do not conform to the recommendations of the
/// Google Objective-C Style Guide. Function names should be in upper camel case
/// including capitalized acronyms and initialisms. Functions that are not of
/// static storage class must also have an appropriate prefix. The function
/// `main` is an exception. Note that this check does not apply to Objective-C
/// method or property declarations.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/google/objc-function-naming.html
class FunctionNamingCheck : public ClangTidyCheck {
public:
  FunctionNamingCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.ObjC;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::google::objc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OBJC_FUNCTION_NAMING_CHECK_H
