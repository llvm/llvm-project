//===--- ContainerDataPointerCheck.h - clang-tidy ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONTAINERDATAPOINTERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONTAINERDATAPOINTERCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {
/// Checks whether a call to `operator[]` and `&` can be replaced with a call to
/// `data()`.
///
/// This only replaces the case where the offset being accessed through the
/// subscript operation is a known constant 0.  This avoids a potential invalid
/// memory access when the container is empty.  Cases where the constant is not
/// explicitly zero can be addressed through the clang static analyzer, and
/// those which cannot be statically identified can be caught using UBSan.
class ContainerDataPointerCheck : public ClangTidyCheck {
public:
  ContainerDataPointerCheck(StringRef Name, ClangTidyContext *Context);

  bool isLanguageVersionSupported(const LangOptions &LO) const override {
    return LO.CPlusPlus11;
  }

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;

  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  const std::vector<StringRef> IgnoredContainers;
};
} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONTAINERDATAPOINTERCHECK_H
