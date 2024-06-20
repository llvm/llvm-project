//===-- Implementation header for bind --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_SOCKET_BIND_H
#define LLVM_LIBC_SRC_SYS_SOCKET_BIND_H

#include <sys/socket.h>

namespace LIBC_NAMESPACE {

int bind(int domain, const struct sockaddr *address, socklen_t address_len);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_SOCKET_BIND_H
