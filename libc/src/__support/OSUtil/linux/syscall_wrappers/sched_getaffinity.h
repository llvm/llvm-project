//===-- Implementation header for sched_getaffinity -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SCHED_GETAFFINITY_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SCHED_GETAFFINITY_H

#include "hdr/types/pid_t.h"
#include "src/__support/CPP/span.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"

#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> sched_getaffinity(pid_t tid,
                                           cpp::span<unsigned char> mask) {
  LIBC_SANITIZER_SYSCALL_PRE(sched_getaffinity, tid, mask.size(), mask.data());
  auto syscall_ret = syscall_checked<int>(SYS_sched_getaffinity, tid,
                                          mask.size(), mask.data());
  LIBC_SANITIZER_SYSCALL_POST(sched_getaffinity,
                              syscall_result_to_long(syscall_ret), tid,
                              mask.size(), mask.data());
  return syscall_ret;
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_SCHED_GETAFFINITY_H
