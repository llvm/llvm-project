//===-- Syscall wrapper for futex -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_FUTEX_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_FUTEX_H

#include "hdr/stdint_proxy.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/OSUtil/linux/syscall.h" // For syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

#if SYS_futex
LIBC_INLINE_VAR constexpr long FUTEX_SYSCALL_NUMBER = SYS_futex;
#elif defined(SYS_futex_time64)
LIBC_INLINE_VAR constexpr long FUTEX_SYSCALL_NUMBER = SYS_futex_time64;
#else
#error "futex and futex_time64 syscalls not available."
#endif

LIBC_INLINE ErrorOr<int> futex(void *futex_addr, int op, uint32_t val,
                               const timespec *timeout, void *futex_addr2,
                               uint32_t val3) {
  return syscall_checked<int>(FUTEX_SYSCALL_NUMBER, futex_addr, op, val,
                              timeout, futex_addr2, val3);
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_FUTEX_H
