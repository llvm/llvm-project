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
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/Models/UncheckedOptionalAccessModel.h"

namespace clang::tidy {

template <> struct OptionEnumMapping<dataflow::SolverMode> {
  static llvm::ArrayRef<std::pair<dataflow::SolverMode, StringRef>>
  getEnumMapping() {
    static constexpr std::pair<dataflow::SolverMode, StringRef> Mapping[] = {
        {dataflow::SolverMode::WatchedLiterals, "watched"},
        {dataflow::SolverMode::Z3, "z3"},
        {dataflow::SolverMode::Tiered, "tiered"},
    };
    return {Mapping};
  }
};

namespace bugprone {

/// Warns when the code is unwrapping a `std::optional<T>`, `absl::optional<T>`,
/// or `base::std::optional<T>` object without assuring that it contains a
/// value.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/unchecked-optional-access.html
class UncheckedOptionalAccessCheck : public ClangTidyCheck {
public:
  UncheckedOptionalAccessCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        ModelOptions{Options.get("IgnoreSmartPointerDereference", false),
                     Options.get("IgnoreValueCalls", false)},
        SolverMode(Options.get("SolverMode", dataflow::SolverMode::Tiered)) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override {
    Options.store(Opts, "IgnoreSmartPointerDereference",
                  ModelOptions.IgnoreSmartPointerDereference);
    Options.store(Opts, "IgnoreValueCalls", ModelOptions.IgnoreValueCalls);
    Options.store(Opts, "SolverMode", SolverMode);
  }

private:
  dataflow::UncheckedOptionalAccessModelOptions ModelOptions;
  dataflow::SolverMode SolverMode;
};

} // namespace bugprone
} // namespace clang::tidy

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNCHECKEDOPTIONALACCESSCHECK_H
