//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_ASSERTEQUALSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_ASSERTEQUALSCHECK_H

#include "../ClangTidyCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang::tidy::objc {

/// Warns if XCTAssertEqual() or XCTAssertNotEqual() is used with at least one
/// operands of type NSString*.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/objc/assert-equals.html
class AssertEqualsCheck final : public ClangTidyCheck {
public:
  AssertEqualsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.ObjC;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::objc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_ASSERTEQUALSCHECK_H
