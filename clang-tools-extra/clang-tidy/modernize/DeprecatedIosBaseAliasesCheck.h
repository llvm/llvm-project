//===--- DeprecatedIosBaseAliasesCheck.h - clang-tidy------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_DEPRECATEDIOSBASEALIASESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_DEPRECATEDIOSBASEALIASESCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace modernize {

/// This check warns the uses of the deprecated member types of ``std::ios_base``
/// and replaces those that have a non-deprecated equivalent.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-deprecated-ios-base-aliases.html
class DeprecatedIosBaseAliasesCheck : public ClangTidyCheck {
public:
  DeprecatedIosBaseAliasesCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_DEPRECATEDIOSBASEALIASESCHECK_H
