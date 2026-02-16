//===--- AvoidPassingMlirOpAsRefCheck.h - clang-tidy ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_AVOIDPASSINGMLIROPASREFCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_AVOIDPASSINGMLIROPASREFCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::llvm_check {

/// Flags function parameters of a type derived from mlir::Op that are passed by
/// reference.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/llvm/avoid-passing-mlir-op-as-ref.html
class AvoidPassingMlirOpAsRefCheck : public ClangTidyCheck {
public:
  AvoidPassingMlirOpAsRefCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::llvm_check

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_AVOIDPASSINGMLIROPASREFCHECK_H
