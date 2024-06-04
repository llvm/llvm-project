//===-- Implementation of at_quick_exit -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/at_quick_exit.h"
#include "src/__support/CPP/mutex.h" // lock_guard
#include "src/__support/blockstore.h"
#include "src/__support/common.h"
#include "src/__support/fixedvector.h"
#include "src/__support/threads/mutex.h"
#include "src/stdlib/exit_handler.h"
namespace LIBC_NAMESPACE {

extern "C" {

int __cxa_at_quick_exit(AtExitCallback *callback, void *payload, void *) {
  return add_atexit_unit(at_quick_exit_callbacks, {callback, payload});
}

void __cxa_finalize_quick_exit(void *dso) {
  if (!dso)
    call_exit_callbacks(at_quick_exit_callbacks);
}

} // extern "C"

LLVM_LIBC_FUNCTION(int, at_quick_exit, (StdCAtExitCallback * callback)) {
  return add_atexit_unit(
      at_quick_exit_callbacks,
      {&stdc_at_exit_func, reinterpret_cast<void *>(callback)});
}

} // namespace LIBC_NAMESPACE
