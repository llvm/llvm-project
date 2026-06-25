//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of usleep.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/usleep.h"
#include "hdr/errno_macros.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/nanosleep.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, usleep, (useconds_t usec)) {
  // POSIX.1-2001 specifies that usleep should fail with EINVAL if the
  // argument is greater than or equal to 1,000,000.
  if (usec >= 1000000) {
    libc_errno = EINVAL;
    return -1;
  }
  if (usec == 0)
    return 0;

  timespec req = {0, static_cast<long>(usec * 1000)};
  auto result = linux_syscalls::nanosleep(&req, nullptr);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
