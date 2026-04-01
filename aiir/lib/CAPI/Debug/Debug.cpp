//===- Debug.cpp - C Interface for AIIR/LLVM Debugging Functions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Debug.h"
#include "aiir-c/Support.h"

#include "aiir/CAPI/Support.h"

#include "llvm/Support/Debug.h"

void aiirEnableGlobalDebug(bool enable) { llvm::DebugFlag = enable; }

bool aiirIsGlobalDebugEnabled() { return llvm::DebugFlag; }

void aiirSetGlobalDebugType(const char *type) {
  // Depending on the NDEBUG flag, this name can be either a function or a macro
  // that expands to something that isn't a funciton call, so we cannot
  // explicitly prefix it with `llvm::` or declare `using` it.
  using namespace llvm;
  setCurrentDebugType(type);
}

void aiirSetGlobalDebugTypes(const char **types, intptr_t n) {
  using namespace llvm;
  setCurrentDebugTypes(types, n);
}

bool aiirIsCurrentDebugType(const char *type) {
  using namespace llvm;
  return isCurrentDebugType(type);
}
