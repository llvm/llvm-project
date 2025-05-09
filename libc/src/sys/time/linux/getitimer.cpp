//===-- Implementation file for getitimer ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/time/getitimer.h"
#include "hdr/types/struct_itimerval.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getitimer, (int which, struct itimerval *curr_value)) {
  long ret =
      LIBC_NAMESPACE::syscall_impl<long>(SYS_getitimer, which, curr_value);
  // On failure, return -1 and set errno.
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
