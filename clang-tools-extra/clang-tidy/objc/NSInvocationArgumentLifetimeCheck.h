//===--- NSInvocationArgumentLifetimeCheck.h - clang-tidy -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_NSINVOCATIONARGUMENTLIFETIMECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_NSINVOCATIONARGUMENTLIFETIMECHECK_H

#include "../ClangTidyCheck.h"
#include "clang/Basic/LangStandard.h"

namespace clang {
namespace tidy {
namespace objc {

/// Finds calls to NSInvocation methods under ARC that don't have proper
/// argument object lifetimes.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/objc/nsinvocation-argument-lifetime.html
class NSInvocationArgumentLifetimeCheck : public ClangTidyCheck {
public:
  NSInvocationArgumentLifetimeCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.ObjC && LangOpts.ObjCAutoRefCount;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace objc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_NSINVOCATIONARGUMENTLIFETIMECHECK_H
