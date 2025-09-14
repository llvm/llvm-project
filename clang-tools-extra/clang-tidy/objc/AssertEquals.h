//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef THIRD_PARTY_LLVM_LLVM_PROJECT_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_OBJCASSERTEQUALS_H_
#define THIRD_PARTY_LLVM_LLVM_PROJECT_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_OBJCASSERTEQUALS_H_

#include "../ClangTidyCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang::tidy::objc {

/// Warn if XCTAssertEqual() or XCTAssertNotEqual() is used with at least one
/// operands of type NSString*.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/objc/assert-equals.html
class AssertEquals final : public ClangTidyCheck {
public:
  AssertEquals(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.ObjC;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::objc

#endif // THIRD_PARTY_LLVM_LLVM_PROJECT_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_OBJCASSERTEQUALS_H_
