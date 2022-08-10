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
#include "clang/Analysis/CallGraph.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/StringSet.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Checker for signal handler functions.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/signal-handler-check.html
class SignalHandlerCheck : public ClangTidyCheck {
public:
  enum class AsyncSafeFunctionSetKind { Minimal, POSIX };

  SignalHandlerCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  /// Check if a function is allowed as a signal handler.
  /// Should test the properties of the function, and check in the code body.
  /// Should not check function calls in the code (this part is done by the call
  /// graph scan).
  /// Bug reports are generated for the whole code body (no stop at the first
  /// found issue). For issues that are not in the code body, only one
  /// bug report is generated.
  /// \param FD The function to check. It may or may not have a definition.
  /// \param CallOrRef Location of the call to this function (in another
  /// function) or the reference to the function (if it is used as a registered
  /// signal handler). This is the location where diagnostics are to be placed.
  /// \param ChainReporter A function that adds bug report notes to display the
  /// chain of called functions from signal handler registration to the current
  /// function. This is called at every generated bug report.
  /// The bool parameter is used like \c SkipPathEnd in \c reportHandlerChain .
  /// \return Returns true if a diagnostic was emitted for this function.
  bool checkFunction(const FunctionDecl *FD, const Expr *CallOrRef,
                     std::function<void(bool)> ChainReporter);
  /// Similar as \c checkFunction but only check for C++14 rules.
  bool checkFunctionCPP14(const FunctionDecl *FD, const Expr *CallOrRef,
                          std::function<void(bool)> ChainReporter);
  /// Returns true if a standard library function is considered
  /// asynchronous-safe.
  bool isStandardFunctionAsyncSafe(const FunctionDecl *FD) const;
  /// Add diagnostic notes to show the call chain of functions from a signal
  /// handler to a function that is called (directly or indirectly) from it.
  /// Also add a note to the place where the signal handler is registered.
  /// @param Itr Position during a call graph depth-first iteration. It contains
  /// the "path" (call chain) from the signal handler to the actual found
  /// function call.
  /// @param HandlerRef Reference to the signal handler function where it is
  /// registered as signal handler.
  /// @param SkipPathEnd If true the last item of the call chain (farthest away
  /// from the \c signal call) is omitted from note generation.
  void reportHandlerChain(const llvm::df_iterator<clang::CallGraphNode *> &Itr,
                          const DeclRefExpr *HandlerRef, bool SkipPathEnd);

  clang::CallGraph CG;

  AsyncSafeFunctionSetKind AsyncSafeFunctionSet;
  llvm::StringSet<> ConformingFunctions;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SIGNALHANDLERCHECK_H
