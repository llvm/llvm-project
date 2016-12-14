//===--- NoMallocCheck.h - clang-tidy----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_NO_MALLOC_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_NO_MALLOC_H

#include "../ClangTidy.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// This checker is concerned with C-style memory management and suggest modern
/// alternatives to it.
/// The check is only enabled in C++. For analyzing malloc calls see Clang
/// Static Analyzer - unix.Malloc.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-no-malloc.html
class NoMallocCheck : public ClangTidyCheck {
public:
  NoMallocCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

  /// Registering for malloc, calloc, realloc and free calls.
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;

  /// Checks matched function calls and gives suggestion to modernize the code.
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_NO_MALLOC_H
