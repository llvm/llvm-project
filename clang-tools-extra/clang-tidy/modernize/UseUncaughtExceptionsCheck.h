//===--- UseUncaughtExceptionsCheck.h - clang-tidy------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_UNCAUGHT_EXCEPTIONS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_UNCAUGHT_EXCEPTIONS_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace modernize {

/// This check will warn on calls to std::uncaught_exception and replace them with calls to
/// std::uncaught_exceptions, since std::uncaught_exception was deprecated in C++17. In case of
/// macro ID there will be only a warning without fixits.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-uncaught-exceptions.html
class UseUncaughtExceptionsCheck : public ClangTidyCheck {
public:
  UseUncaughtExceptionsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_UNCAUGHT_EXCEPTIONS_H
