//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of prctl.
///
//===----------------------------------------------------------------------===//

#include "src/sys/prctl/prctl.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/prctl.h"
#include "src/__support/common.h"

#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, prctl, (int option, ...)) {
  va_list vargs;
  va_start(vargs, option);
  // Try to consume (and pass to the syscall) the maximum possible number of
  // arguments, even though for some "option" values only few will be provided.
  unsigned long arg2 = va_arg(vargs, unsigned long);
  unsigned long arg3 = va_arg(vargs, unsigned long);
  unsigned long arg4 = va_arg(vargs, unsigned long);
  unsigned long arg5 = va_arg(vargs, unsigned long);
  va_end(vargs);
  auto result = linux_syscalls::prctl(option, arg2, arg3, arg4, arg5);
  // The manpage states that "... return the nonnegative values described
  // above. All other option values return 0 on success. On error,
  // -1 is returned, and errno is set to indicate the error."
  // According to the kernel implementation
  // (https://github.com/torvalds/linux/blob/bee0e7762ad2c6025b9f5245c040fcc36ef2bde8/kernel/sys.c#L2442),
  // return value from the syscall is set to 0 on default so we do not need to
  // set the value on success manually.
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return static_cast<int>(result.value());
}

} // namespace LIBC_NAMESPACE_DECL
