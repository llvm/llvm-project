//===--- ReplaceMemcpyWithStdCopy.h - clang-tidy------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACE_WITH_STDCOPY_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACE_WITH_STDCOPY_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/Basic/LangOptions.h"

namespace clang::tidy::modernize {

// Replace C-style calls to functions like memmove and memcpy with analogous
// calls to std::copy or std::copy_n, depending on the context
class ReplaceWithStdCopyCheck : public ClangTidyCheck {
public:
  ReplaceWithStdCopyCheck(StringRef Name, ClangTidyContext *Context);
  ~ReplaceWithStdCopyCheck() override = default;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void storeOptions(ClangTidyOptions::OptionMap &Options) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void tryIssueFixIt(const ast_matchers::MatchFinder::MatchResult &Result,
                     const DiagnosticBuilder &Diag, const CallExpr &CallNode);

  utils::IncludeInserter Inserter;

  const bool FlagMemcpy;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACE_WITH_STDCOPY_H
