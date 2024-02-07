//===-- Implementation header for connect -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_SOCKET_RECV_H
#define LLVM_LIBC_SRC_SYS_SOCKET_RECV_H

#include <sys/socket.h>

namespace LIBC_NAMESPACE {

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_SOCKET_RECV_H
