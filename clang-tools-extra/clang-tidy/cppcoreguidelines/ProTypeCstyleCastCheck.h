//===--- ProTypeCstyleCastCheck.h - clang-tidy-------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_CSTYLE_CAST_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_CSTYLE_CAST_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// This check flags all use of C-style casts that perform a static_cast
/// downcast, const_cast, or reinterpret_cast.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-pro-type-cstyle-cast.html
class ProTypeCstyleCastCheck : public ClangTidyCheck {
public:
  ProTypeCstyleCastCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_CSTYLE_CAST_H
