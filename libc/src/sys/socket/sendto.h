//===-- Implementation header for sendto ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_SOCKET_SENDTO_H
#define LLVM_LIBC_SRC_SYS_SOCKET_SENDTO_H

#include <sys/socket.h>

namespace LIBC_NAMESPACE {

ssize_t sendto(int sockfd, const void *buf, size_t len, int flags,
               const struct sockaddr *dest_addr, socklen_t addrlen);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_SOCKET_SENDTO_H
