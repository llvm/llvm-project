//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of the recvmmsg function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_SOCKET_RECVMMSG_H
#define LLVM_LIBC_SRC_SYS_SOCKET_RECVMMSG_H

#include "hdr/types/struct_mmsghdr.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int recvmmsg(int sockfd, struct mmsghdr *msgvec, unsigned int vlen, int flags,
             struct timespec *timeout);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_SOCKET_RECVMMSG_H
