//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for truncate.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_TRUNCATE_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_TRUNCATE_H

#include "hdr/stdint_proxy.h"
#include "hdr/types/off_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> truncate(const char *path, off_t len) {
#ifdef SYS_truncate64
  // Prefer truncate64 to preserve large offsets on 32-bit systems.
  LIBC_SANITIZER_SYSCALL_PRE(truncate64, path, len,
                             static_cast<uint64_t>(len) >> 32);
  auto syscall_ret = syscall_checked<int>(SYS_truncate64, path, len,
                                          static_cast<uint64_t>(len) >> 32);
  LIBC_SANITIZER_SYSCALL_POST(truncate64, syscall_result_to_long(syscall_ret),
                              path, len, static_cast<uint64_t>(len) >> 32);
  return syscall_ret;
#elif defined(SYS_truncate)
  LIBC_SANITIZER_SYSCALL_PRE(truncate, path, len);
  auto syscall_ret = syscall_checked<int>(SYS_truncate, path, len);
  LIBC_SANITIZER_SYSCALL_POST(truncate, syscall_result_to_long(syscall_ret),
                              path, len);
  return syscall_ret;
#else
#error "truncate and truncate64 syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_TRUNCATE_H
