//===--- ReferenceToConstructedTemporaryCheck.h - clang-tidy ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REFERENCETOCONSTRUCTEDTEMPORARYCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REFERENCETOCONSTRUCTEDTEMPORARYCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Detects C++ code where a reference variable is used to extend the lifetime
/// of a temporary object that has just been constructed.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/reference-to-constructed-temporary.html
class ReferenceToConstructedTemporaryCheck : public ClangTidyCheck {
public:
  ReferenceToConstructedTemporaryCheck(StringRef Name,
                                       ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
  std::optional<TraversalKind> getCheckTraversalKind() const override;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REFERENCETOCONSTRUCTEDTEMPORARYCHECK_H
