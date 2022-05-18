//===--- StaticAccessedThroughInstanceCheck.h - clang-tidy-------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_STATIC_ACCESSED_THROUGH_INSTANCE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_STATIC_ACCESSED_THROUGH_INSTANCE_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace readability {

/// Checks for member expressions that access static members through
/// instances and replaces them with uses of the appropriate qualified-id.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/static-accessed-through-instance.html
class StaticAccessedThroughInstanceCheck : public ClangTidyCheck {
public:
  StaticAccessedThroughInstanceCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        NameSpecifierNestingThreshold(
            Options.get("NameSpecifierNestingThreshold", 3U)) {}

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  llvm::Optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  const unsigned NameSpecifierNestingThreshold;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_STATIC_ACCESSED_THROUGH_INSTANCE_H
