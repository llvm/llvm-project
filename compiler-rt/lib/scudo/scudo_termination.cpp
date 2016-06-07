//===-- scudo_termination.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// This file contains bare-bones termination functions to replace the
/// __sanitizer ones, in order to avoid any potential abuse of the callbacks
/// functionality.
///
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_common.h"

namespace __sanitizer {

bool AddDieCallback(DieCallbackType callback) { return true; }

bool RemoveDieCallback(DieCallbackType callback) { return true; }

void SetUserDieCallback(DieCallbackType callback) {}

void NORETURN Die() {
  if (common_flags()->abort_on_error)
    Abort();
  internal__exit(common_flags()->exitcode);
}

void SetCheckFailedCallback(CheckFailedCallbackType callback) {}

void NORETURN CheckFailed(const char *file, int line, const char *cond,
                          u64 v1, u64 v2) {
  Report("Sanitizer CHECK failed: %s:%d %s (%lld, %lld)\n", file, line, cond,
                                                            v1, v2);
  Die();
}

} // namespace __sanitizer
