//===--- FunctionSizeCheck.h - clang-tidy -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONSIZECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONSIZECHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// Checks for large functions based on various metrics.
///
/// These options are supported:
///
///   * `LineThreshold` - flag functions exceeding this number of lines. The
///     default is `-1` (ignore the number of lines).
///   * `StatementThreshold` - flag functions exceeding this number of
///     statements. This may differ significantly from the number of lines for
///     macro-heavy code. The default is `800`.
///   * `BranchThreshold` - flag functions exceeding this number of control
///     statements. The default is `-1` (ignore the number of branches).
///   * `ParameterThreshold` - flag functions having a high number of parameters.
///     The default is `6`.
class FunctionSizeCheck : public ClangTidyCheck {
public:
  FunctionSizeCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const unsigned LineThreshold;
  const unsigned StatementThreshold;
  const unsigned BranchThreshold;
  const unsigned ParameterThreshold;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONSIZECHECK_H
