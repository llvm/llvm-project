//===-- Implementation of atexit ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/atexit.h"
#include "src/__support/CPP/blockstore.h"
#include "src/__support/common.h"
#include "src/__support/threads/mutex.h"

namespace __llvm_libc {

namespace {

Mutex handler_list_mtx(false, false, false);

using AtExitCallback = void(void);
using ExitCallbackList = cpp::ReverseOrderBlockStore<AtExitCallback *, 32>;
constinit ExitCallbackList exit_callbacks;

} // namespace

namespace internal {

void call_exit_callbacks() {
  handler_list_mtx.lock();
  while (!exit_callbacks.empty()) {
    auto *callback = exit_callbacks.back();
    exit_callbacks.pop_back();
    handler_list_mtx.unlock();
    callback();
    handler_list_mtx.lock();
  }
  ExitCallbackList::destroy(&exit_callbacks);
}

} // namespace internal

LLVM_LIBC_FUNCTION(int, atexit, (AtExitCallback * callback)) {
  handler_list_mtx.lock();
  exit_callbacks.push_back(callback);
  handler_list_mtx.unlock();
  return 0;
}

} // namespace __llvm_libc
