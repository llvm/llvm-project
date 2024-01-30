//===-- Implementation header for epoll_wait function -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_EPOLL_EPOLL_WAIT_H
#define LLVM_LIBC_SRC_SYS_EPOLL_EPOLL_WAIT_H

// TODO: Use this include once the include headers are also using quotes.
// #include "include/llvm-libc-types/struct_epoll_event.h"

#include <sys/epoll.h>

namespace LIBC_NAMESPACE {

int epoll_wait(int epfd, epoll_event *events, int maxevents, int timeout);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_EPOLL_EPOLL_WAIT_H
