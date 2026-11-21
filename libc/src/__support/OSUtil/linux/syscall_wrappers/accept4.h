//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for accept4.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_ACCEPT4_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_ACCEPT4_H

#include "hdr/types/socklen_t.h"
#include "hdr/types/struct_sockaddr.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> accept4(int sockfd, struct sockaddr *addr,
                                 socklen_t *addrlen, int flags) {
  LIBC_SANITIZER_SYSCALL_PRE(accept4, sockfd, addr, addrlen, flags);
  auto syscall_ret =
      syscall_checked<int>(SYS_accept4, sockfd, addr, addrlen, flags);
  LIBC_SANITIZER_SYSCALL_POST(accept4, syscall_result_to_long(syscall_ret),
                              sockfd, addr, addrlen, flags);
  return syscall_ret;
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_ACCEPT4_H
