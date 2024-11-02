//===--- Implementation of exit_handler------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/exit_handler.h"
#include "src/__support/CPP/mutex.h" // lock_guard

namespace LIBC_NAMESPACE {

constinit ExitCallbackList at_quick_exit_callbacks;
constinit ExitCallbackList atexit_callbacks;

Mutex handler_list_mtx(false, false, false, false);

void stdc_at_exit_func(void *payload) {
  reinterpret_cast<StdCAtExitCallback *>(payload)();
}

void call_exit_callbacks(ExitCallbackList &callbacks) {
  handler_list_mtx.lock();
  while (!callbacks.empty()) {
    AtExitUnit &unit = callbacks.back();
    callbacks.pop_back();
    handler_list_mtx.unlock();
    unit.callback(unit.payload);
    handler_list_mtx.lock();
  }
  ExitCallbackList::destroy(&callbacks);
}

int add_atexit_unit(ExitCallbackList &callbacks, const AtExitUnit &unit) {
  cpp::lock_guard lock(handler_list_mtx);
  if (callbacks.push_back(unit))
    return 0;
  return -1;
}

} // namespace LIBC_NAMESPACE
