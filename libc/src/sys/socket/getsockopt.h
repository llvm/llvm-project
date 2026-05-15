//===-- Implementation header for getsockopt --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_SOCKET_GETSOCKOPT_H
#define LLVM_LIBC_SRC_SYS_SOCKET_GETSOCKOPT_H

#include "src/__support/macros/config.h"

#include "hdr/types/socklen_t.h"

namespace LIBC_NAMESPACE_DECL {

int getsockopt(int sockfd, int level, int optname, void *__restrict optval,
               socklen_t *__restrict optlen);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_SOCKET_GETSOCKOPT_H
