//===---------- Linux implementation of the Linux pkey_alloc function -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/pkey_alloc.h"

#include "hdr/errno_macros.h"             // For ENOSYS
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pkey_alloc,
                   (unsigned int flags, unsigned int access_rights)) {
#if !defined(SYS_pkey_alloc)
  libc_errno = ENOSYS;
  return -1;
#else
  int ret =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_pkey_alloc, flags, access_rights);
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return static_cast<int>(ret);
#endif
}

} // namespace LIBC_NAMESPACE_DECL
