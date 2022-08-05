//===--- UseNodiscardCheck.h - clang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USENODISCARDCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USENODISCARDCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace modernize {

/// Add ``[[nodiscard]]`` to non-void const-member functions with no
/// arguments or pass-by-value or pass by const-reference arguments.
/// \code
///    bool empty() const;
///    bool empty(const Bar &) const;
///    bool empty(int bar) const;
/// \endcode
/// Is converted to:
/// \code
///    [[nodiscard]] bool empty() const;
///    [[nodiscard]] bool empty(const Bar &) const;
///    [[nodiscard]] bool empty(int bar) const;
/// \endcode
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/use-nodiscard.html
class UseNodiscardCheck : public ClangTidyCheck {
public:
  UseNodiscardCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const std::string NoDiscardMacro;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USENODISCARDCHECK_H
