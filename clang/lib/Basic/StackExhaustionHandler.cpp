//===--- StackExhaustionHandler.cpp -  - A utility for warning once when close
// to out of stack space -------*- C++ -*-===//
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

#include "clang/Basic/StackExhaustionHandler.h"
#include "clang/Basic/Stack.h"

void clang::StackExhaustionHandler::runWithSufficientStackSpace(
    SourceLocation Loc, llvm::function_ref<void()> Fn) {
  clang::runWithSufficientStackSpace([&] { warnStackExhausted(Loc); }, Fn);
}

void clang::StackExhaustionHandler::warnOnStackNearlyExhausted(
    SourceLocation Loc) {
  if (isStackNearlyExhausted())
    warnStackExhausted(Loc);
}

void clang::StackExhaustionHandler::warnStackExhausted(SourceLocation Loc) {
  // Only warn about this once.
  if (!WarnedStackExhausted) {
    DiagsRef.Report(Loc, diag::warn_stack_exhausted);
    WarnedStackExhausted = true;
  }
}
