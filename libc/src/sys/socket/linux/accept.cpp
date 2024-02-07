//===-- Linux implementation of accept ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/accept.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/errno/libc_errno.h"

#include <linux/net.h>   // For SYS_SOCKET socketcall number.
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, accept,
                   (int domain, sockaddr *__restrict address,
                    socklen_t *__restrict address_len)) {
#ifdef SYS_accept
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_accept, domain, address,
                                              address_len);
#elif defined(SYS_socketcall)
  unsigned long sockcall_args[3] = {static_cast<unsigned long>(domain),
                                    reinterpret_cast<unsigned long>(address),
                                    static_cast<unsigned long>(address_len)};
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_socketcall, SYS_ACCEPT,
                                              sockcall_args);
#else
#error "socket and socketcall syscalls unavailable for this platform."
#endif
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE
