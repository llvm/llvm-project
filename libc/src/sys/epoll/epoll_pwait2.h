//===-- Implementation header for epoll_pwait2 function ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_EPOLL_EPOLL_PWAIT2_H
#define LLVM_LIBC_SRC_SYS_EPOLL_EPOLL_PWAIT2_H

#include <signal.h>    // For sigset_t
#include <sys/epoll.h> // For epoll_event
#include <time.h>      // For timespec

namespace LIBC_NAMESPACE {

// TODO: sigmask and timeout should be nullable
int epoll_pwait2(int epfd, struct epoll_event *events, int maxevents,
                 const struct timespec *timeout, const sigset_t *sigmask);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_EPOLL_EPOLL_PWAIT2_H
