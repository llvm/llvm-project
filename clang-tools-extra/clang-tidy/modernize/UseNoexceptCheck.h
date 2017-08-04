//===--- UseNoexceptCheck.h - clang-tidy-------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_NOEXCEPT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_NOEXCEPT_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace modernize {

/// \brief Replace dynamic exception specifications, with
/// `noexcept` (or user-defined macro) or `noexcept(false)`.
/// \code
///   void foo() throw();
///   void bar() throw(int);
/// \endcode
/// Is converted to:
/// \code
///   void foo() ;
///   void bar() noexcept(false);
/// \endcode
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-noexcept.html
class UseNoexceptCheck : public ClangTidyCheck {
public:
  UseNoexceptCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const std::string NoexceptMacro;
  bool UseNoexceptFalse;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_NOEXCEPT_H
