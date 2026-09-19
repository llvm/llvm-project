//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_INEFFICIENTCONTAINERASSIGNMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_INEFFICIENTCONTAINERASSIGNMENTCHECK_H

#include "../ClangTidyCheck.h"
#include <vector>

namespace clang::tidy::performance {

/// Finds assignments of a freshly constructed temporary container to a
/// container of the same type and suggests the `assign` member function, or
/// an equivalent in-place rewrite, that reuses the existing storage.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/performance/inefficient-container-assignment.html
class InefficientContainerAssignmentCheck : public ClangTidyCheck {
public:
  InefficientContainerAssignmentCheck(StringRef Name,
                                      ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

private:
  const std::vector<StringRef> ContainerClasses;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_INEFFICIENTCONTAINERASSIGNMENTCHECK_H
