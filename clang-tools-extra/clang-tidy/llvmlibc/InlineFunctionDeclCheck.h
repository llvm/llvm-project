//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVMLIBC_INLINEFUNCTIONDECLCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVMLIBC_INLINEFUNCTIONDECLCHECK_H

#include "../ClangTidyCheck.h"
#include "../FileExtensionsSet.h"

namespace clang::tidy::llvm_libc {

/// Checks that explicitly and implicitly inline functions in headers files
/// are tagged with the LIBC_INLINE macro.
///
/// For more information about the LIBC_INLINE macro, see
/// https://libc.llvm.org/dev/code_style.html.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/llvmlibc/inline-function-decl-check.html
class InlineFunctionDeclCheck : public ClangTidyCheck {
public:
  InlineFunctionDeclCheck(StringRef Name, ClangTidyContext *Context);

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  // Ignore implicit functions (e.g. implicit constructors or destructors)
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  FileExtensionsSet HeaderFileExtensions;
};

} // namespace clang::tidy::llvm_libc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVMLIBC_INLINEFUNCTIONDECLCHECK_H
