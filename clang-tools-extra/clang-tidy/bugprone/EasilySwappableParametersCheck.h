//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_EASILYSWAPPABLEPARAMETERSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_EASILYSWAPPABLEPARAMETERSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Finds function definitions where parameters of convertible types follow
/// each other directly, making call sites prone to calling the function with
/// swapped (or badly ordered) arguments.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/easily-swappable-parameters.html
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
  const std::vector<StringRef> IgnoredParameterNames;

  /// The parameter typename suffixes (as written in the source code) to be
  /// ignored.
  const std::vector<StringRef> IgnoredParameterTypeSuffixes;

  /// Whether to consider differently qualified versions of the same type
  /// mixable.
  const bool QualifiersMix;

  /// Whether to model implicit conversions "in full" (conditions apply)
  /// during analysis and consider types that are implicitly convertible to
  /// one another mixable.
  const bool ModelImplicitConversions;

  /// If enabled, diagnostics for parameters that are used together in a
  /// similar way are not emitted.
  const bool SuppressParametersUsedTogether;

  /// The number of characters two parameter names might be dissimilar at
  /// either end for the report about the parameters to be silenced.
  /// E.g. the names "LHS" and "RHS" are 1-dissimilar suffixes of each other,
  /// while "Text1" and "Text2" are 1-dissimilar prefixes of each other.
  const std::size_t NamePrefixSuffixSilenceDissimilarityThreshold;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_EASILYSWAPPABLEPARAMETERSCHECK_H
