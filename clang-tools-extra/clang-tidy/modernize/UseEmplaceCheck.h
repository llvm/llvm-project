//===----------------------------------------------------------------------===//
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

namespace clang::tidy::modernize {

/// This check looks for cases when inserting new element into std::vector but
/// the element is constructed temporarily.
/// It replaces those calls for emplace_back of arguments passed to
/// constructor of temporary object.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/use-emplace.html
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
  const std::vector<StringRef> ContainersWithPushBack;
  const std::vector<StringRef> ContainersWithPush;
  const std::vector<StringRef> ContainersWithPushFront;
  const std::vector<StringRef> SmartPointers;
  const std::vector<StringRef> TupleTypes;
  const std::vector<StringRef> TupleMakeFunctions;
  const std::vector<StringRef> EmplacyFunctions;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EMPLACE_H
