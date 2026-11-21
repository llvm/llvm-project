//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for ftruncate.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_FTRUNCATE_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_FTRUNCATE_H

#include "hdr/types/off_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> ftruncate(int fd, off_t len) {
#ifdef SYS_ftruncate
  LIBC_SANITIZER_SYSCALL_PRE(ftruncate, fd, len);
  auto syscall_ret = syscall_checked<int>(SYS_ftruncate, fd, len);
  LIBC_SANITIZER_SYSCALL_POST(ftruncate, syscall_result_to_long(syscall_ret),
                              fd, len);
  return syscall_ret;
#elif defined(SYS_ftruncate64)
  // Same as ftruncate but can handle large offsets on 32-bit systems.
  static_assert(sizeof(off_t) == 8);
  LIBC_SANITIZER_SYSCALL_PRE(ftruncate64, fd, (long)len,
                             (long)(((uint64_t)(len)) >> 32));
  auto syscall_ret = syscall_checked<int>(SYS_ftruncate64, fd, (long)len,
                                          (long)(((uint64_t)(len)) >> 32));
  LIBC_SANITIZER_SYSCALL_POST(ftruncate64, syscall_result_to_long(syscall_ret),
                              fd, (long)len, (long)(((uint64_t)(len)) >> 32));
  return syscall_ret;
#else
#error "ftruncate and ftruncate64 syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_FTRUNCATE_H
