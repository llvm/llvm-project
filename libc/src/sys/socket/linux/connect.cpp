//===-- Linux implementation of connect
//--------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/connect.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"
#include <linux/net.h>   // For SYS_SOCKET socketcall number.
#include <sys/socket.h>  // For the types
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, connect,
                   (int sockfd, const struct sockaddr *addr,
                    socklen_t addrlen)) {
#ifdef SYS_connect
  int ret =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_connect, sockfd, addr, addrlen);
#elif defined(SYS_socketcall)
  unsigned long sockcall_args[4] = {static_cast<unsigned long>(sockfd),
                                    reinterpret_cast<unsigned long>(addr),
                                    static_cast<unsigned long>(addrlen)};
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_socketcall, SYS_CONNECT,
                                              sockcall_args);
#else
#error "socket and socketcall syscalls unavailable for this platform."
#endif
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE
