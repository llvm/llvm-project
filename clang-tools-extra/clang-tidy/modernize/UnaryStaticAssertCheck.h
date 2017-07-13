//===--- UnaryStaticAssertCheck.h - clang-tidy-------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_UNARY_STATIC_ASSERT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_UNARY_STATIC_ASSERT_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace modernize {

/// Replaces a static_assert declaration with an empty message
/// with the unary version.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-unary-static-assert.html
class UnaryStaticAssertCheck : public ClangTidyCheck {
public:
  UnaryStaticAssertCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_UNARY_STATIC_ASSERT_H
