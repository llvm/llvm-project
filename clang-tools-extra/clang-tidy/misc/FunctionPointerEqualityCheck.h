//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_FUNCTIONPOINTEREQUALITYCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_FUNCTIONPOINTEREQUALITYCHECK_H

#include "../ClangTidyCheck.h"
#include <vector>

namespace clang::tidy::misc {

/// Detects equality comparisons between function pointers and detects
/// associative or unordered containers with function pointer key types, which
/// can be unsound when aggressive Identical Code Folding (``--icf=all``) is
/// enabled.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/misc/function-pointer-equality.html
class FunctionPointerEqualityCheck : public ClangTidyCheck {
public:
  FunctionPointerEqualityCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  /// Semicolon-separated fully-qualified container names (e.g.
  /// ``::std::set;::std::map``) to warn on the first template argument for
  /// being a function pointer type key.
  const std::vector<StringRef> KeyedContainerTypes;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_FUNCTIONPOINTEREQUALITYCHECK_H
