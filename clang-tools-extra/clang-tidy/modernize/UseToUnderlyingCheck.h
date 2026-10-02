//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USETOUNDERLYINGCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USETOUNDERLYINGCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"
#include <optional>

namespace clang::tidy::modernize {

/// Finds casts from a scoped enumeration to its underlying integer type and
/// replaces them with a call to ``std::to_underlying`` (C++23).
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/modernize/use-to-underlying.html
class UseToUnderlyingCheck : public ClangTidyCheck {
public:
  /// How to treat an imprecise cast, i.e. a cast whose destination type is an
  /// integer type other than the enumeration's underlying type.
  enum class ImpreciseCastsKind {
    /// Do not diagnose imprecise casts.
    Ignore,
    /// Diagnose imprecise casts but do not offer a fix-it.
    Warn,
    /// Wrap the operand in a call to the replacement function, preserving the
    /// original destination type.
    PreserveType,
    /// Replace the whole cast with a call to the replacement function, changing
    /// the resulting type to the underlying type.
    UseUnderlyingType,
  };

  UseToUnderlyingCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  const ImpreciseCastsKind ImpreciseCasts;
  const StringRef ReplacementFunction;
  StringRef ReplacementFunctionHeader;
  utils::IncludeInserter IncludeInserter;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USETOUNDERLYINGCHECK_H
