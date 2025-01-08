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
        ModelOptions{Options.get("IgnoreSmartPointerDereference", false)},
        IgnoreTestTus(Options.get("IgnoreTestTUs", false)) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onStartOfTranslationUnit() override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override {
    Options.store(Opts, "IgnoreSmartPointerDereference",
                  ModelOptions.IgnoreSmartPointerDereference);
    Options.store(Opts, "IgnoreTestTUs", IgnoreTestTus);
  }

private:
  dataflow::UncheckedOptionalAccessModelOptions ModelOptions;

  // Tracks the Option of whether we should ignore test TUs (e.g., googletest,
  // catch2). Currently we have many false positives in tests, making it
  // difficult to find true positives and developers end up ignoring the
  // warnings in tests, reducing the check's effectiveness.
  // Reasons for false positives (once fixed we could remove this option):
  // - has_value() checks wrapped in googletest assertion macros are not handled
  //   (e.g., EXPECT_TRUE() and fall through, or ASSERT_TRUE() and crash,
  //    or more complex ones like ASSERT_THAT(x, Not(Eq(std::nullopt))))
  //   Catch2's REQUIRE, CHECK, etc. General macro issue:
  //   https://github.com/llvm/llvm-project/issues/62600
  // - we don't handle state carried over from test fixture constructors/setup
  //   to test case bodies (constructor may initialize an optional to a value)
  // - developers may make shortcuts in tests making assumptions and
  //   use the test runs (expecially with sanitizers) to check assumptions.
  bool IgnoreTestTus = false;

  // Records whether the current TU includes the test-specific headers (e.g.,
  // googletest, catch2), in which case we assume it is a test TU.
  // This along with `IgnoreTestTus` allows us to disable checking in test TUs.
  bool IsTestTu = false;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNCHECKEDOPTIONALACCESSCHECK_H
