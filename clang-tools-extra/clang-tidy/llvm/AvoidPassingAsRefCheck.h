//===--- AvoidPassingAsRefCheck.h - clang-tidy ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_AVOIDPASSINGASREFCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_AVOIDPASSINGASREFCHECK_H

#include "../ClangTidyCheck.h"
#include <string>
#include <vector>

namespace clang::tidy::llvm_check {

/// Flags function parameters of types that should be passed by value but are
/// passed by reference instead.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/llvm/avoid-passing-as-ref.html
class AvoidPassingAsRefCheck : public ClangTidyCheck {
public:
  AvoidPassingAsRefCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const std::string ClassNames;
  const std::vector<StringRef> ClassNameList;
};

} // namespace clang::tidy::llvm_check

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_AVOIDPASSINGASREFCHECK_H
