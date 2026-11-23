//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of timerfd_gettime.
///
//===----------------------------------------------------------------------===//

#include "src/sys/timerfd/timerfd_gettime.h"

#include "hdr/types/struct_itimerspec.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/timerfd_gettime.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, timerfd_gettime,
                   (int fd, struct itimerspec *curr_value)) {
  ErrorOr<int> ret = linux_syscalls::timerfd_gettime(fd, curr_value);
  if (!ret) {
    libc_errno = ret.error();
    return -1;
  }
  return *ret;
}

} // namespace LIBC_NAMESPACE_DECL
