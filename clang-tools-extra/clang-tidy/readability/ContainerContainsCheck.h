//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONTAINERCONTAINSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONTAINERCONTAINSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Finds usages of `container.count()` and
/// `container.find() == container.end()` which should be replaced by a call
/// to the `container.contains()` method.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/container-contains.html
class ContainerContainsCheck : public ClangTidyCheck {
public:
  ContainerContainsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) final;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) final;
  bool isLanguageVersionSupported(const LangOptions &LO) const final {
    return LO.CPlusPlus;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_AsIs;
  }
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONTAINERCONTAINSCHECK_H
