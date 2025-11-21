//===----------------------------------------------------------------------===//
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

namespace clang::tidy::modernize {

using NamespaceName = llvm::SmallString<40>;

class NS : public llvm::SmallVector<const NamespaceDecl *, 6> {
public:
  std::optional<SourceRange>
  getCleanedNamespaceFrontRange(const SourceManager &SM,
                                const LangOptions &LangOpts) const;
  SourceRange getReplacedNamespaceFrontRange() const;
  SourceRange getNamespaceBackRange(const SourceManager &SM,
                                    const LangOptions &LangOpts) const;
  SourceRange getDefaultNamespaceBackRange() const;
  void appendName(NamespaceName &Str) const;
  void appendCloseComment(NamespaceName &Str) const;
};

class ConcatNestedNamespacesCheck : public ClangTidyCheck {
public:
  ConcatNestedNamespacesCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool unsupportedNamespace(const NamespaceDecl &ND, bool IsChild) const;
  bool singleNamedNamespaceChild(const NamespaceDecl &ND) const;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus17;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  using NamespaceContextVec = llvm::SmallVector<NS, 6>;

  void reportDiagnostic(const SourceManager &SM, const LangOptions &LangOpts);
  NamespaceContextVec Namespaces;
};
} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_CONCATNESTEDNAMESPACESCHECK_H
