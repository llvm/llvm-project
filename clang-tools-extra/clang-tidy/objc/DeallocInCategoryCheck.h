//===--- DeallocInCategoryCheck.h - clang-tidy ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_DEALLOCINCATEGORYCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_DEALLOCINCATEGORYCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace objc {

/// Finds implementations of -dealloc in Objective-C categories. The category
/// implementation will override any dealloc in the class implementation,
/// potentially causing issues.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/objc-dealloc-in-category.html
class DeallocInCategoryCheck final : public ClangTidyCheck {
public:
  DeallocInCategoryCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.ObjC;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace objc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_DEALLOCINCATEGORYCHECK_H
