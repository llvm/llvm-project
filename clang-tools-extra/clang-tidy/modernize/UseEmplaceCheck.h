//===--- UseEmplaceCheck.h - clang-tidy--------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EMPLACE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EMPLACE_H

#include "../ClangTidyCheck.h"
#include <string>
#include <vector>

namespace clang {
namespace tidy {
namespace modernize {

/// This check looks for cases when inserting new element into std::vector but
/// the element is constructed temporarily.
/// It replaces those calls for emplace_back of arguments passed to
/// constructor of temporary object.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-emplace.html
class UseEmplaceCheck : public ClangTidyCheck {
public:
  UseEmplaceCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const bool IgnoreImplicitConstructors;
  const std::vector<std::string> ContainersWithPushBack;
  const std::vector<std::string> SmartPointers;
  const std::vector<std::string> TupleTypes;
  const std::vector<std::string> TupleMakeFunctions;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EMPLACE_H
