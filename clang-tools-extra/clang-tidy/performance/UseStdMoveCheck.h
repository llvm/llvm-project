//===--- UseStdMoveCheck.h - clang-tidy -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_USESTDMOVECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_USESTDMOVECHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace clang::tidy::performance {

/// Finds copy construction and assignment on the last use of a variable.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/performance/use-std-move.html
class UseStdMoveCheck : public ClangTidyCheck {
public:
  UseStdMoveCheck(StringRef Name, ClangTidyContext *Context);
  ~UseStdMoveCheck() override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void onEndOfTranslationUnit() override;

private:
  struct FunctionAnalysis;
  FunctionAnalysis *getFunctionAnalysis(const FunctionDecl *FD,
                                        ASTContext *Context);
  llvm::DenseMap<const FunctionDecl *, std::unique_ptr<FunctionAnalysis>>
      AnalysisCache;
  std::set<std::pair<unsigned, std::string>> Diagnosed;
  utils::IncludeInserter Inserter;
  const std::vector<StringRef> AllowedTypes;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_USESTDMOVECHECK_H
