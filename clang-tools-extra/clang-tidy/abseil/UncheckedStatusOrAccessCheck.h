#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_UNCHECKEDSTATUSORACCESSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_UNCHECKEDSTATUSORACCESSCHECK_H

#include "../ClangTidyCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang::tidy::abseil {

// Warns when the code is unwrapping an absl::StatusOr<T> object without
// assuring that it contains a value.
//
// For details on the dataflow analysis implemented in this check see:
// clang/lib/Analysis/FlowSensitive/Models/UncheckedStatusOrAccessModel.cpp
class UncheckedStatusOrAccessCheck : public ClangTidyCheck {
public:
  using ClangTidyCheck::ClangTidyCheck;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
};

} // namespace clang::tidy::abseil

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_UNCHECKEDSTATUSORACCESSCHECK_H
