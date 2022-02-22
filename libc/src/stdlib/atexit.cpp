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
#include "src/threads/mtx_init.h"
#include "src/threads/mtx_lock.h"
#include "src/threads/mtx_unlock.h"

namespace __llvm_libc {

namespace {

mtx_t lock;
// TODO need an easier way to use mtx_t internally, or use pthread_mutex_t
// with PTHREAD_MUTEX_INITIALIZER when it lands.
struct Init {
  Init() { __llvm_libc::mtx_init(&lock, mtx_plain); }
} init;

// TOOD should we make cpp::vector like llvm::SmallVector<T, N> where it will
// allocate at least N before needing dynamic allocation?
static cpp::vector<void (*)(void)> handlers;

} // namespace

namespace internal {

void call_exit_handlers() {
  __llvm_libc::mtx_lock(&lock);
  // TODO: implement rbegin() + rend() for cpp::vector
  for (int i = handlers.size() - 1; i >= 0; i--) {
    __llvm_libc::mtx_unlock(&lock);
    handlers[i]();
    __llvm_libc::mtx_lock(&lock);
  }
}

} // namespace internal

LLVM_LIBC_FUNCTION(int, atexit, (void (*function)())) {
  __llvm_libc::mtx_lock(&lock);
  handlers.push_back(function);
  __llvm_libc::mtx_unlock(&lock);
  return 0;
}

} // namespace __llvm_libc
