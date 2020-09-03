//===--- GlobalVariableDeclarationCheck.h - clang-tidy-----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OBJC_GLOBAL_VARIABLE_DECLARATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OBJC_GLOBAL_VARIABLE_DECLARATION_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace google {
namespace objc {

/// The check for Objective-C global variables and constants naming convention.
/// The declaration should follow the patterns of 'k[A-Z].*' (constants) or
/// 'g[A-Z].*' (variables).
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/google-objc-global-variable-declaration.html
class GlobalVariableDeclarationCheck : public ClangTidyCheck {
 public:
  GlobalVariableDeclarationCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.ObjC;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

}  // namespace objc
}  // namespace google
}  // namespace tidy
}  // namespace clang

#endif  // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OBJC_GLOBAL_VARIABLE_DECLARATION_H
