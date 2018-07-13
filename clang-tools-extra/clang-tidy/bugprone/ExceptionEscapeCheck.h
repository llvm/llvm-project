//===--- ExceptionEscapeCheck.h - clang-tidy---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_EXCEPTION_ESCAPE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_EXCEPTION_ESCAPE_H

#include "../ClangTidy.h"

#include "llvm/ADT/StringSet.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Finds functions which should not throw exceptions: Destructors, move
/// constructors, move assignment operators, the main() function,
/// swap() functions, functions marked with throw() or noexcept and functions
/// given as option to the checker.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone-exception-escape.html
class ExceptionEscapeCheck : public ClangTidyCheck {
public:
  ExceptionEscapeCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  std::string RawFunctionsThatShouldNotThrow;
  std::string RawIgnoredExceptions;

  llvm::StringSet<> FunctionsThatShouldNotThrow;
  llvm::StringSet<> IgnoredExceptions;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_EXCEPTION_ESCAPE_H
