//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for getsockopt.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_GETSOCKOPT_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_GETSOCKOPT_H

#include "hdr/types/socklen_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> getsockopt(int sockfd, int level, int optname,
                                    void *optval, socklen_t *optlen) {
  LIBC_SANITIZER_SYSCALL_PRE(getsockopt, sockfd, level, optname, optval,
                             optlen);
  auto syscall_ret = syscall_checked<int>(SYS_getsockopt, sockfd, level,
                                          optname, optval, optlen);
  LIBC_SANITIZER_SYSCALL_POST(getsockopt, syscall_result_to_long(syscall_ret),
                              sockfd, level, optname, optval, optlen);
  return syscall_ret;
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_GETSOCKOPT_H
