//===-- Implementation header for connect -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_SOCKET_CONNECT_H
#define LLVM_LIBC_SRC_SYS_SOCKET_CONNECT_H

#include "hdr/types/socklen_t.h"
#include "hdr/types/struct_sockaddr.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_SOCKET_CONNECT_H
