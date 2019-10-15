//===--- NotNullTerminatedResultCheck.h - clang-tidy ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_NOT_NULL_TERMINATED_RESULT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_NOT_NULL_TERMINATED_RESULT_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Finds function calls where it is possible to cause a not null-terminated
/// result. Usually the proper length of a string is 'strlen(src) + 1' or
/// equal length of this expression, because the null terminator needs an extra
/// space. Without the null terminator it can result in undefined behaviour
/// when the string is read.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone-not-null-terminated-result.html
class NotNullTerminatedResultCheck : public ClangTidyCheck {
public:
  NotNullTerminatedResultCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;

private:
  // If non-zero it is specifying if the target environment is considered to
  // implement '_s' suffixed memory and string handler functions which are safer
  // than older version (e.g. 'memcpy_s()'). The default value is '1'.
  const int WantToUseSafeFunctions;

  bool UseSafeFunctions = false;

  void memoryHandlerFunctionFix(
      StringRef Name, const ast_matchers::MatchFinder::MatchResult &Result);
  void memcpyFix(StringRef Name,
                 const ast_matchers::MatchFinder::MatchResult &Result,
                 DiagnosticBuilder &Diag);
  void memcpy_sFix(StringRef Name,
                   const ast_matchers::MatchFinder::MatchResult &Result,
                   DiagnosticBuilder &Diag);
  void memchrFix(StringRef Name,
                 const ast_matchers::MatchFinder::MatchResult &Result);
  void memmoveFix(StringRef Name,
                  const ast_matchers::MatchFinder::MatchResult &Result,
                  DiagnosticBuilder &Diag);
  void strerror_sFix(const ast_matchers::MatchFinder::MatchResult &Result);
  void ncmpFix(StringRef Name,
               const ast_matchers::MatchFinder::MatchResult &Result);
  void xfrmFix(StringRef Name,
               const ast_matchers::MatchFinder::MatchResult &Result);
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_NOT_NULL_TERMINATED_RESULT_H
