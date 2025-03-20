//===-- Linux implementation of sendto ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/sendto.h"

#include <linux/net.h>   // For SYS_SOCKET socketcall number.
#include <sys/syscall.h> // For syscall numbers.

#include "hdr/types/socklen_t.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_sockaddr.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, sendto,
                   (int sockfd, const void *buf, size_t len, int flags,
                    const struct sockaddr *dest_addr, socklen_t addrlen)) {
#ifdef SYS_sendto
  ssize_t ret = LIBC_NAMESPACE::syscall_impl<ssize_t>(
      SYS_sendto, sockfd, buf, len, flags, dest_addr, addrlen);
#elif defined(SYS_socketcall)
  unsigned long sockcall_args[6] = {static_cast<unsigned long>(sockfd),
                                    reinterpret_cast<unsigned long>(buf),
                                    static_cast<unsigned long>(len),
                                    static_cast<unsigned long>(flags),
                                    reinterpret_cast<unsigned long>(dest_addr),
                                    static_cast<unsigned long>(addrlen)};
  ssize_t ret = LIBC_NAMESPACE::syscall_impl<ssize_t>(
      SYS_socketcall, SYS_SENDTO, sockcall_args);
#else
#error "socket and socketcall syscalls unavailable for this platform."
#endif
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
