//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_USESTDPRINTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_USESTDPRINTCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"

namespace clang::tidy::modernize {
/// Convert calls to printf-like functions to std::print and std::println
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/use-std-print.html
class UseStdPrintCheck : public ClangTidyCheck {
public:
  UseStdPrintCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    if (ReplacementPrintFunction == "std::print" ||
        ReplacementPrintlnFunction == "std::println")
      return LangOpts.CPlusPlus23;
    return LangOpts.CPlusPlus;
  }
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  Preprocessor *PP;
  bool StrictMode;
  std::vector<StringRef> PrintfLikeFunctions;
  std::vector<StringRef> FprintfLikeFunctions;
  StringRef ReplacementPrintFunction;
  StringRef ReplacementPrintlnFunction;
  utils::IncludeInserter IncludeInserter;
  std::optional<StringRef> MaybeHeaderToInclude;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_USESTDPRINTCHECK_H
