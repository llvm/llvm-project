//===--- ReplaceMemcpyWithStdCopy.h - clang-tidy------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACE_MEMCPY_WITH_STDCOPY_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACE_MEMCPY_WITH_STDCOPY_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"
#include <memory>
#include <string>
#include <vector>

namespace clang {
namespace tidy {
namespace modernize {

// Replace the C memcpy function with std::copy
class ReplaceMemcpyWithStdCopy : public ClangTidyCheck {
public:
  ReplaceMemcpyWithStdCopy(StringRef Name, ClangTidyContext *Context);
  ~ReplaceMemcpyWithStdCopy() override = default;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Options) override;

private:
  void renameFunction(DiagnosticBuilder &Diag, const CallExpr *MemcpyNode);
  void reorderArgs(DiagnosticBuilder &Diag, const CallExpr *MemcpyNode);
  void insertHeader(DiagnosticBuilder &Diag, const CallExpr *MemcpyNode,
                    SourceManager *const SM);

private:
  std::unique_ptr<utils::IncludeInserter> Inserter;
  utils::IncludeInserter IncludeInserter;
  const utils::IncludeSorter::IncludeStyle IncludeStyle;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACE_MEMCPY_WITH_STDCOPY_H
