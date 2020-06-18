//===-- Linux specific definitions for threads implementations. --*- C++ -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_LINUX_THREAD_UTILS_H
#define LLVM_LIBC_SRC_THREADS_LINUX_THREAD_UTILS_H

#include "thread_start_args.h"

#include <stdatomic.h>
#include <stdint.h>

namespace __llvm_libc {

// The futex data has to be exactly 4 bytes long. However, we use a uint type
// here as we do not want to use `_Atomic uint32_t` as the _Atomic keyword which
// is C only. The header stdatomic.h does not define an atomic type
// corresponding to `uint32_t` or to something which is exactly 4 bytes wide.
using FutexData = atomic_uint;

// We use a tri-state mutex because we want to avoid making syscalls
// as much as possible. In `mtx_unlock` a syscall to wake waiting threads is
// made only if the mutex status is `MutexStatus::Waiting`.
enum MutexStatus : uint32_t { MS_Free, MS_Locked, MS_Waiting };

static_assert(sizeof(atomic_uint) == 4,
              "Size of the `atomic_uint` type is not 4 bytes on your platform. "
              "The implementation of the standard threads library for linux "
              "requires that size of `atomic_uint` be 4 bytes.");

struct ThreadParams {
  static constexpr uintptr_t DefaultStackSize = 1 << 16; // 64 KB
  static constexpr uint32_t ClearTIDValue = 0xABCD1234;
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_THREADS_LINUX_THREAD_UTILS_H
