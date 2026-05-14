//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for getsockname.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_SOCKET_GETSOCKNAME_H
#define LLVM_LIBC_SRC_SYS_SOCKET_GETSOCKNAME_H

#include "hdr/types/socklen_t.h"
#include "hdr/types/struct_sockaddr.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int getsockname(int sockfd, struct sockaddr *__restrict addr,
                socklen_t *__restrict addrlen);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_SOCKET_GETSOCKNAME_H
