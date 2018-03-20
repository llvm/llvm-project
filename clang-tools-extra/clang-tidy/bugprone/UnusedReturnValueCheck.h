//===--- UnusedReturnValueCheck.h - clang-tidy-------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNUSEDRETURNVALUECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNUSEDRETURNVALUECHECK_H

#include "../ClangTidy.h"
#include <string>

namespace clang {
namespace tidy {
namespace bugprone {

/// Detects function calls where the return value is unused.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone-unused-return-value.html
class UnusedReturnValueCheck : public ClangTidyCheck {
public:
  UnusedReturnValueCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  std::string CheckedFunctions;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNUSEDRETURNVALUECHECK_H
