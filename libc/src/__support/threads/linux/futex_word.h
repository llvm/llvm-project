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
#include <sys/syscall.h>

namespace __llvm_libc {

// Futexes are 32 bits in size on all platforms, including 64-bit platforms.
using FutexWordType = uint32_t;

#if SYS_futex
constexpr auto FUTEX_SYSCALL_ID = SYS_futex;
#elif defined(SYS_futex_time64)
constexpr auto FUTEX_SYSCALL_ID = SYS_futex_time64;
#else
#error "futex and futex_time64 syscalls not available."
#endif

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_THREADS_LINUX_FUTEX_WORD_H
