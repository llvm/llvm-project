//===--- StaticallyConstructedObjectsCheck.h - clang-tidy--------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_STATICALLY_CONSTRUCTED_OBJECTS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_STATICALLY_CONSTRUCTED_OBJECTS_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace fuchsia {

/// Constructing global, non-trivial objects with static storage is
/// disallowed, unless the object is statically initialized with a constexpr 
/// constructor or has no explicit constructor.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/fuchsia/statically-constructed-objects.html
class StaticallyConstructedObjectsCheck : public ClangTidyCheck {
public:
  StaticallyConstructedObjectsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace fuchsia
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_STATICALLY_CONSTRUCTED_OBJECTS_H
