//===--- UncheckedOptionalAccessCheck.h - clang-tidy ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNCHECKEDOPTIONALACCESSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNCHECKEDOPTIONALACCESSCHECK_H

#include "../ClangTidyCheck.h"
#include "../ClangTidyOptions.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/FlowSensitive/Models/UncheckedOptionalAccessModel.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"

namespace clang::tidy::bugprone {

/// Warns when the code is unwrapping a `std::optional<T>`, `absl::optional<T>`,
/// or `base::std::optional<T>` object without assuring that it contains a
/// value.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/unchecked-optional-access.html
class UncheckedOptionalAccessCheck : public ClangTidyCheck {
public:
  UncheckedOptionalAccessCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        ModelOptions{
            Options.getLocalOrGlobal("IgnoreSmartPointerDereference", false)},
        ignore_test_tus_(Options.getLocalOrGlobal("IgnoreTestTUs", false)) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onStartOfTranslationUnit() override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override {
    Options.store(Opts, "IgnoreSmartPointerDereference",
                  ModelOptions.IgnoreSmartPointerDereference);
    Options.store(Opts, "IgnoreTestTUs", ignore_test_tus_);
  }

private:
  dataflow::UncheckedOptionalAccessModelOptions ModelOptions;

  // Tracks the Option of whether we should ignore test TUs (e.g., googletest).
  // Currently we have many false positives in tests, making it difficult to
  // find true positives and developers end up ignoring the warnings in tests,
  // reducing the check's effectiveness.
  // Reasons for false positives (once fixed we could remove this option):
  // - has_value() checks wrapped in googletest assertion macros are not handled
  //   (e.g., EXPECT_TRUE() and fall through, or ASSERT_TRUE() and crash,
  //    or more complex ones like ASSERT_THAT(x, Not(Eq(std::nullopt))))
  // - we don't handle state carried over from test fixture constructors/setup
  //   to test case bodies (constructor may initialize an optional to a value)
  // - developers may make shortcuts in tests making assumptions and
  //   use the test runs (expecially with sanitizers) to check assumptions.
  //   This is different from production code in that test code should have
  //   near 100% coverage (if not covered by the test itself, it is dead code).
  bool ignore_test_tus_ = false;

  // Records whether the current TU includes the test-specific headers (e.g.,
  // googletest), in which case we assume it is a test TU of some sort.
  // This along with the setting `ignore_test_tus_` allows us to disable
  // checking for all test TUs.
  bool is_test_tu_ = false;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNCHECKEDOPTIONALACCESSCHECK_H
