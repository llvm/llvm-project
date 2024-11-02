//===--- Definition of a type for a futex word ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_THREADS_LINUX_FUTEX_WORD_H
#define LLVM_LIBC_SRC_SUPPORT_THREADS_LINUX_FUTEX_WORD_H

#include <stdint.h>

namespace __llvm_libc {

// Futexes are 32 bits in size on all platforms, including 64-bit platforms.
using FutexWordType = uint32_t;

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_THREADS_LINUX_FUTEX_WORD_H
