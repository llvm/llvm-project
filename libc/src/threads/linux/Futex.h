//===-- Linux futex related definitions -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_LINUX_FUTEX_H
#define LLVM_LIBC_SRC_THREADS_LINUX_FUTEX_H

#include <stdatomic.h>

namespace __llvm_libc {

// The futex data has to be exactly 4 bytes long. However, we use a uint type
// here as we do not want to use `_Atomic uint32_t` as the _Atomic keyword which
// is C only. The header stdatomic.h does not define an atomic type
// corresponding to `uint32_t` or to something which is exactly 4 bytes wide.
using FutexWord = atomic_uint;
static_assert(sizeof(atomic_uint) == 4,
              "Size of the `atomic_uint` type is not 4 bytes on your platform. "
              "The implementation of the standard threads library for linux "
              "requires that size of `atomic_uint` be 4 bytes.");

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_THREADS_LINUX_FUTEX_H
