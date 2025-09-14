//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNCHECKEDOPTIONALACCESSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNCHECKEDOPTIONALACCESSCHECK_H

#include "../ClangTidyCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/FlowSensitive/Models/UncheckedOptionalAccessModel.h"

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
        ModelOptions{Options.get("IgnoreSmartPointerDereference", false)} {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override {
    Options.store(Opts, "IgnoreSmartPointerDereference",
                  ModelOptions.IgnoreSmartPointerDereference);
  }

private:
  dataflow::UncheckedOptionalAccessModelOptions ModelOptions;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNCHECKEDOPTIONALACCESSCHECK_H
