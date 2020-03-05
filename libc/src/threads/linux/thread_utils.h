//===--- Linux specific definitions to support mutex operations --*- C++ -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_LINUX_THREAD_UTILS_H
#define LLVM_LIBC_SRC_THREADS_LINUX_THREAD_UTILS_H

#include <stdint.h>

using FutexData = _Atomic uint32_t;

struct ThreadParams {
  static constexpr uintptr_t DefaultStackSize = 1 << 15; // 32 KB
  static constexpr uint32_t ClearTIDValue = 0xABCD1234;
};

#endif // LLVM_LIBC_SRC_THREADS_LINUX_THREAD_UTILS_H
