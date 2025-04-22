//===--- StackExhaustionHandler.h - A utility for warning once when close to out
// of stack space -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines a utilitiy for warning once when close to out of stack space.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_STACK_EXHAUSTION_HANDLER_H
#define LLVM_CLANG_BASIC_STACK_EXHAUSTION_HANDLER_H

#include "clang/Basic/Diagnostic.h"

namespace clang {
class StackExhaustionHandler {
public:
  StackExhaustionHandler(DiagnosticsEngine &diags) : DiagsRef(diags) {}

  /// Run some code with "sufficient" stack space. (Currently, at least 256K
  /// is guaranteed). Produces a warning if we're low on stack space and
  /// allocates more in that case. Use this in code that may recurse deeply to
  /// avoid stack overflow.
  void runWithSufficientStackSpace(SourceLocation Loc,
                                   llvm::function_ref<void()> Fn);

  /// Check to see if we're low on stack space and produce a warning if we're
  /// low on stack space (Currently, at least 256Kis guaranteed).
  void warnOnStackNearlyExhausted(SourceLocation Loc);

private:
  /// Warn that the stack is nearly exhausted.
  void warnStackExhausted(SourceLocation Loc);

  DiagnosticsEngine &DiagsRef;
  bool WarnedStackExhausted = false;
};
} // end namespace clang

#endif // LLVM_CLANG_BASIC_STACK_EXHAUSTION_HANDLER_H
