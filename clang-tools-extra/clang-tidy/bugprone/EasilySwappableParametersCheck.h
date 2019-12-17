//===--- EasilySwappableParametersCheck.h - clang-tidy ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_EASILYSWAPPABLEPARAMETERSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_EASILYSWAPPABLEPARAMETERSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Finds function definitions where parameters of convertible types follow
/// each other directly, making call sites prone to calling the function with
/// swapped (or badly ordered) arguments.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone-easily-swappable-parameters.html
class EasilySwappableParametersCheck : public ClangTidyCheck {
public:
  EasilySwappableParametersCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  /// The minimum length of an adjacent swappable parameter range required for
  /// a diagnostic.
  const std::size_t MinimumLength;

  /// The parameter names (as written in the source text) to be ignored.
  const std::vector<std::string> IgnoredParameterNames;

  /// The parameter typename suffixes (as written in the source code) to be
  /// ignored.
  const std::vector<std::string> IgnoredParameterTypeSuffixes;

  /// Whether to consider differently qualified versions of the same type
  /// mixable.
  const bool QualifiersMix;

  /// Whether to model implicit conversions "in full" (conditions apply)
  /// during analysis and consider types that are implicitly convertible to
  /// one another mixable.
  const bool ModelImplicitConversions;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_EASILYSWAPPABLEPARAMETERSCHECK_H
