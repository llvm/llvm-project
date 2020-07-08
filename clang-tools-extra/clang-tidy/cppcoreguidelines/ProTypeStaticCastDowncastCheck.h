//===--- ProTypeStaticCastDowncastCheck.h - clang-tidy-----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_STATIC_CAST_DOWNCAST_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_STATIC_CAST_DOWNCAST_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// Checks for usages of static_cast, where a base class is downcasted to a
/// derived class.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-pro-type-static-cast-downcast.html
class ProTypeStaticCastDowncastCheck : public ClangTidyCheck {
public:
  ProTypeStaticCastDowncastCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_STATIC_CAST_DOWNCAST_H
