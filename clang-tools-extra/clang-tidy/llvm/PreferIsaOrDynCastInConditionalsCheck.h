//===--- PreferIsaOrDynCastInConditionalsCheck.h - clang-tidy ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_PREFERISAORDYNCASTINCONDITIONALSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_PREFERISAORDYNCASTINCONDITIONALSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace llvm_check {

/// Looks at conditionals and finds and replaces cases of ``cast<>``, which will
/// assert rather than return a null pointer, and ``dyn_cast<>`` where
/// the return value is not captured.  Additionally, finds and replaces cases that match the
/// pattern ``var && isa<X>(var)``, where ``var`` is evaluated twice.
///
/// Finds cases like these:
/// \code
///  if (auto x = cast<X>(y)) {}
///  // is replaced by:
///  if (auto x = dyn_cast<X>(y)) {}
///
///  if (cast<X>(y)) {}
///  // is replaced by:
///  if (isa<X>(y)) {}
///
///  if (dyn_cast<X>(y)) {}
///  // is replaced by:
///  if (isa<X>(y)) {}
///
///  if (var && isa<T>(var)) {}
///  // is replaced by:
///  if (isa_and_nonnull<T>(var.foo())) {}
/// \endcode
///
///  // Other cases are ignored, e.g.:
/// \code
///  if (auto f = cast<Z>(y)->foo()) {}
///  if (cast<Z>(y)->foo()) {}
///  if (X.cast(y)) {}
/// \endcode
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/llvm-prefer-isa-or-dyn-cast-in-conditionals.html
class PreferIsaOrDynCastInConditionalsCheck : public ClangTidyCheck {
public:
  PreferIsaOrDynCastInConditionalsCheck(StringRef Name,
                                        ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace llvm_check
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_PREFERISAORDYNCASTINCONDITIONALSCHECK_H
