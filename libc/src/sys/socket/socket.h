//===-- Implementation header for socket ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_SOCKET_SOCKET_H
#define LLVM_LIBC_SRC_SYS_SOCKET_SOCKET_H

namespace __llvm_libc {

int socket(int domain, int type, int protocol);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SYS_SOCKET_SOCKET_H
