//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_INCLUDECLEANER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_INCLUDECLEANER_H

#include "../ClangTidyCheck.h"
#include "../ClangTidyDiagnosticConsumer.h"
#include "../ClangTidyOptions.h"
#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/Regex.h"
#include <vector>

namespace clang::tidy::misc {

/// Checks for unused and missing includes. Generates findings only for
/// the main file of a translation unit.
/// Findings correspond to https://clangd.llvm.org/design/include-cleaner.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc/include-cleaner.html
class IncludeCleanerCheck : public ClangTidyCheck {
public:
  IncludeCleanerCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;

private:
  include_cleaner::RecordedPP RecordedPreprocessor;
  include_cleaner::PragmaIncludes RecordedPI;
  const Preprocessor *PP = nullptr;
  std::vector<StringRef> IgnoreHeaders;
  // Whether emit only one finding per usage of a symbol.
  const bool DeduplicateFindings;
  // Whether to report unused includes.
  const bool UnusedIncludes;
  // Whether to report missing includes.
  const bool MissingIncludes;
  llvm::SmallVector<llvm::Regex> IgnoreHeadersRegex;
  bool shouldIgnore(const include_cleaner::Header &H);
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_INCLUDECLEANER_H
