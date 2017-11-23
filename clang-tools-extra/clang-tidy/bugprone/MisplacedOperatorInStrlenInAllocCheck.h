//===--- MisplacedOperatorInStrlenInAllocCheck.h - clang-tidy----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MISPLACED_OPERATOR_IN_STRLEN_IN_ALLOC_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MISPLACED_OPERATOR_IN_STRLEN_IN_ALLOC_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Finds cases where ``1`` is added to the string in the argument to a function
/// in the ``strlen()`` family instead of the result and value is used as an
/// argument to a memory allocation function.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone-misplaced-operator-in-strlen-in-alloc.html
class MisplacedOperatorInStrlenInAllocCheck : public ClangTidyCheck {
public:
  MisplacedOperatorInStrlenInAllocCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MISPLACED_OPERATOR_IN_STRLEN_IN_ALLOC_H
