//===-- Implementation header for epoll_pwait2 function ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_EPOLL_EPOLL_PWAIT2_H
#define LLVM_LIBC_SRC_SYS_EPOLL_EPOLL_PWAIT2_H

// TODO: use our internal sigset_t type (issue #86034)
// #include "include/llvm-libc-types/sigset_t.h"
#include <signal.h>

#include "include/llvm-libc-types/struct_epoll_event.h"
#include "include/llvm-libc-types/struct_timespec.h"

namespace LIBC_NAMESPACE {

// TODO: sigmask and timeout should be nullable
int epoll_pwait2(int epfd, epoll_event *events, int maxevents,
                 const timespec *timeout, const sigset_t *sigmask);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_EPOLL_EPOLL_PWAIT2_H
