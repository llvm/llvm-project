//===-- Implementation of atexit ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/atexit.h"
#include "src/__support/CPP/vector.h"
#include "src/__support/common.h"
#include "src/__support/threads/mutex.h"

namespace __llvm_libc {

namespace {

Mutex handler_list_mtx(false, false, false);

// TOOD should we make cpp::vector like llvm::SmallVector<T, N> where it will
// allocate at least N before needing dynamic allocation?
static cpp::vector<void (*)(void)> handlers;

} // namespace

namespace internal {

void call_exit_handlers() {
  handler_list_mtx.lock();
  // TODO: implement rbegin() + rend() for cpp::vector
  for (int i = handlers.size() - 1; i >= 0; i--) {
    handler_list_mtx.unlock();
    handlers[i]();
    handler_list_mtx.lock();
  }
}

} // namespace internal

LLVM_LIBC_FUNCTION(int, atexit, (void (*function)())) {
  handler_list_mtx.lock();
  handlers.push_back(function);
  handler_list_mtx.unlock();
  return 0;
}

} // namespace __llvm_libc
