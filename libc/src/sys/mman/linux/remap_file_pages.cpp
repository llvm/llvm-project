//===------- Linux implementation of the remap_file_pages function --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/remap_file_pages.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, remap_file_pages,
                   (void *addr, size_t size, int prot, size_t pgoff,
                    int flags)) {
#ifdef SYS_remap_file_pages
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_remap_file_pages,
                                              reinterpret_cast<long>(addr),
                                              size, prot, pgoff, flags);
#else
#error "remap_file_pages syscall is not available."
#endif

  // A negative return value indicates an error with the magnitude of the
  // value being the error code.
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
