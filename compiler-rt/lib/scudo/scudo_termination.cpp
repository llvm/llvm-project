//===-- scudo_termination.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file contains bare-bones termination functions to replace the
/// __sanitizer ones, in order to avoid any potential abuse of the callbacks
/// functionality.
///
//===----------------------------------------------------------------------===//

#include "scudo_utils.h"

#include "sanitizer_common/sanitizer_common.h"

namespace __sanitizer {

bool AddDieCallback(DieCallbackType Callback) { return true; }

bool RemoveDieCallback(DieCallbackType Callback) { return true; }

void SetUserDieCallback(DieCallbackType Callback) {}

void NORETURN Die() {
  if (common_flags()->abort_on_error)
    Abort();
  internal__exit(common_flags()->exitcode);
}

void SetCheckUnwindCallback(void (*callback)()) {}

void NORETURN CheckFailed(const char *File, int Line, const char *Condition,
                          u64 Value1, u64 Value2) {
  __scudo::dieWithMessage("CHECK failed at %s:%d %s (%lld, %lld)\n",
                          File, Line, Condition, Value1, Value2);
}

}  // namespace __sanitizer
