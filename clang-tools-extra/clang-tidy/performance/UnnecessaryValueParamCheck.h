//===--- UnnecessaryValueParamCheck.h - clang-tidy---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_UNNECESSARY_VALUE_PARAM_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_UNNECESSARY_VALUE_PARAM_H

#include "../ClangTidy.h"
#include "../utils/IncludeInserter.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"

namespace clang {
namespace tidy {
namespace performance {

/// \brief A check that flags value parameters of expensive to copy types that
/// can safely be converted to const references.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance-unnecessary-value-param.html
class UnnecessaryValueParamCheck : public ClangTidyCheck {
public:
  UnnecessaryValueParamCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void registerPPCallbacks(CompilerInstance &Compiler) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void onEndOfTranslationUnit() override;

private:
  void handleMoveFix(const ParmVarDecl &Var, const DeclRefExpr &CopyArgument,
                     const ASTContext &Context);

  llvm::DenseMap<const FunctionDecl *, FunctionParmMutationAnalyzer>
      MutationAnalyzers;
  std::unique_ptr<utils::IncludeInserter> Inserter;
  const utils::IncludeSorter::IncludeStyle IncludeStyle;
  const std::vector<std::string> AllowedTypes;
};

} // namespace performance
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_UNNECESSARY_VALUE_PARAM_H
