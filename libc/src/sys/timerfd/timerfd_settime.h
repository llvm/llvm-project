//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for timerfd_settime.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_TIMERFD_TIMERFD_SETTIME_H
#define LLVM_LIBC_SRC_SYS_TIMERFD_TIMERFD_SETTIME_H

#include "hdr/types/struct_itimerspec.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int timerfd_settime(int fd, int flags, const struct itimerspec *new_value,
                    struct itimerspec *old_value);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_TIMERFD_TIMERFD_SETTIME_H
