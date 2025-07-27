//===--- Stack.cpp - Utilities for dealing with stack space ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines utilities for dealing with stack allocation and stack space.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/Stack.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/ProgramStack.h"

static LLVM_THREAD_LOCAL uintptr_t BottomOfStack = 0;

void clang::noteBottomOfStack(bool ForceSet) {
  if (!BottomOfStack || ForceSet)
    BottomOfStack = llvm::getStackPointer();
}

bool clang::isStackNearlyExhausted() {
  // We consider 256 KiB to be sufficient for any code that runs between checks
  // for stack size.
  constexpr size_t SufficientStack = 256 << 10;

  // If we don't know where the bottom of the stack is, hope for the best.
  if (!BottomOfStack)
    return false;

  intptr_t StackDiff =
      (intptr_t)llvm::getStackPointer() - (intptr_t)BottomOfStack;
  size_t StackUsage = (size_t)std::abs(StackDiff);

  // If the stack pointer has a surprising value, we do not understand this
  // stack usage scheme. (Perhaps the target allocates new stack regions on
  // demand for us.) Don't try to guess what's going on.
  if (StackUsage > DesiredStackSize)
    return false;

  return StackUsage >= DesiredStackSize - SufficientStack;
}

void clang::runWithSufficientStackSpaceSlow(llvm::function_ref<void()> Diag,
                                            llvm::function_ref<void()> Fn) {
  llvm::CrashRecoveryContext CRC;
  // Preserve the BottomOfStack in case RunSafelyOnNewStack uses split stacks.
  uintptr_t PrevBottom = BottomOfStack;
  CRC.RunSafelyOnNewStack([&] {
    noteBottomOfStack(true);
    Diag();
    Fn();
  }, DesiredStackSize);
  BottomOfStack = PrevBottom;
}
