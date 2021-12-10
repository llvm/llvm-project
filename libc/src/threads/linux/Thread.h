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

struct ThreadParams {
  static constexpr uintptr_t DEFAULT_STACK_SIZE = 1 << 16; // 64 KB
  static constexpr uint32_t CLEAR_TID_VALUE = 0xABCD1234;
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_THREADS_LINUX_THREAD_UTILS_H
