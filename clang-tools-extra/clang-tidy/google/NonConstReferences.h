//===--- NonConstReferences.h - clang-tidy ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_NON_CONST_REFERENCES_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_NON_CONST_REFERENCES_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace google {
namespace runtime {

/// Checks the usage of non-constant references in function parameters.
///
/// https://google.github.io/styleguide/cppguide.html#Reference_Arguments
class NonConstReferences : public ClangTidyCheck {
public:
  NonConstReferences(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const std::vector<std::string> IncludedTypes;
};

} // namespace runtime
} // namespace google
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_NON_CONST_REFERENCES_H
