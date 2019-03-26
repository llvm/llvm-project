//===--- ConcatNestedNamespacesCheck.h - clang-tidy--------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_CONCATNESTEDNAMESPACESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_CONCATNESTEDNAMESPACESCHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
namespace tidy {
namespace modernize {

class ConcatNestedNamespacesCheck : public ClangTidyCheck {
public:
  ConcatNestedNamespacesCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  using NamespaceContextVec = llvm::SmallVector<const NamespaceDecl *, 6>;
  using NamespaceString = llvm::SmallString<40>;

  void reportDiagnostic(const SourceRange &FrontReplacement,
                        const SourceRange &BackReplacement);
  NamespaceString concatNamespaces();
  NamespaceContextVec Namespaces;
};
} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_CONCATNESTEDNAMESPACESCHECK_H
