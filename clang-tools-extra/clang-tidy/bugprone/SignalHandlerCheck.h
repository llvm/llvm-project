//===--- SignalHandlerCheck.h - clang-tidy ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SIGNALHANDLERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SIGNALHANDLERCHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/StringSet.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Checker for signal handler functions.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone-signal-handler-check.html
class SignalHandlerCheck : public ClangTidyCheck {
public:
  SignalHandlerCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void reportBug(const FunctionDecl *CalledFunction, const Expr *CallOrRef,
                 const CallExpr *SignalCall, const FunctionDecl *HandlerDecl);
  bool isSystemCallAllowed(const FunctionDecl *FD) const;

  static llvm::StringSet<> StrictConformingFunctions;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SIGNALHANDLERCHECK_H
