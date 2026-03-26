//===---------- Linux implementation of the prctl function ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/prctl/prctl.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.

#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, prctl,
                   (int option, unsigned long arg2, unsigned long arg3,
                    unsigned long arg4, unsigned long arg5)) {
  long ret =
      LIBC_NAMESPACE::syscall_impl(SYS_prctl, option, arg2, arg3, arg4, arg5);
  // The manpage states that "... return the nonnegative values described
  // above. All other option values return 0 on success. On error,
  // -1 is returned, and errno is set to indicate the error."
  // According to the kernel implementation
  // (https://github.com/torvalds/linux/blob/bee0e7762ad2c6025b9f5245c040fcc36ef2bde8/kernel/sys.c#L2442),
  // return value from the syscall is set to 0 on default so we do not need to
  // set the value on success manually.
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  return static_cast<int>(ret);
}

} // namespace LIBC_NAMESPACE_DECL
