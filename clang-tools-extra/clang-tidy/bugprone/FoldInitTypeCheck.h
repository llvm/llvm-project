//===--- FoldInitTypeCheck.h - clang-tidy------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_FOLD_INIT_TYPE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_FOLD_INIT_TYPE_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Find and flag invalid initializer values in folds, e.g. std::accumulate.
/// Example:
/// \code
///   auto v = {65536L * 65536 * 65536};
///   std::accumulate(begin(v), end(v), 0 /* int type is too small */);
/// \endcode
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone-fold-init-type.html
class FoldInitTypeCheck : public ClangTidyCheck {
public:
  FoldInitTypeCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void doCheck(const BuiltinType &IterValueType, const BuiltinType &InitType,
               const ASTContext &Context, const CallExpr &CallNode);
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_FOLD_INIT_TYPE_H
