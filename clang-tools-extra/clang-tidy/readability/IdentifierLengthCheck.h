//===--- IdentifierLengthCheck.h - clang-tidy ---------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IDENTIFIERLENGTHCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IDENTIFIERLENGTHCHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/Support/Regex.h"

namespace clang {
namespace tidy {
namespace readability {

/// Warns about identifiers names whose length is too short.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-identifier-length.html
class IdentifierLengthCheck : public ClangTidyCheck {
public:
  IdentifierLengthCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const unsigned MinimumVariableNameLength;
  const unsigned MinimumLoopCounterNameLength;
  const unsigned MinimumExceptionNameLength;
  const unsigned MinimumParameterNameLength;

  std::string IgnoredVariableNamesInput;
  llvm::Regex IgnoredVariableNames;

  std::string IgnoredLoopCounterNamesInput;
  llvm::Regex IgnoredLoopCounterNames;

  std::string IgnoredExceptionVariableNamesInput;
  llvm::Regex IgnoredExceptionVariableNames;

  std::string IgnoredParameterNamesInput;
  llvm::Regex IgnoredParameterNames;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_IDENTIFIERLENGTHCHECK_H
