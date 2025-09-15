//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_USEINTERNALLINKAGECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_USEINTERNALLINKAGECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::misc {

/// Detects variables and functions that can be marked as static or moved into
/// an anonymous namespace to enforce internal linkage.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc/use-internal-linkage.html
class UseInternalLinkageCheck : public ClangTidyCheck {
public:
  UseInternalLinkageCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

  enum class FixModeKind {
    None,
    UseStatic,
  };

private:
  FileExtensionsSet HeaderFileExtensions;
  FixModeKind FixMode;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_USEINTERNALLINKAGECHECK_H
