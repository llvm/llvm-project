//===--- UseEqualsDefaultCheck.h - clang-tidy--------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EQUALS_DEFAULT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EQUALS_DEFAULT_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace modernize {

/// \brief Replace default bodies of special member functions with '= default;'.
/// \code
///   struct A {
///     A() {}
///     ~A();
///   };
///   A::~A() {}
/// \endcode
/// Is converted to:
/// \code
///   struct A {
///     A() = default;
///     ~A();
///   };
///   A::~A() = default;
/// \endcode
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-equals-default.html
class UseEqualsDefaultCheck : public ClangTidyCheck {
public:
  UseEqualsDefaultCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EQUALS_DEFAULT_H
