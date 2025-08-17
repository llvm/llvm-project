//===---------- Linux implementation of the POSIX mmap function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/mmap.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <linux/param.h> // For EXEC_PAGESIZE.
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

// This function is currently linux only. It has to be refactored suitably if
// mmap is to be supported on non-linux operating systems also.
LLVM_LIBC_FUNCTION(void *, mmap,
                   (void *addr, size_t size, int prot, int flags, int fd,
                    off_t offset)) {
  // A lot of POSIX standard prescribed validation of the parameters is not
  // done in this function as modern linux versions do it in the syscall.
  // TODO: Perform argument validation not done by the linux syscall.

  // EXEC_PAGESIZE is used for the page size. While this is OK for x86_64, it
  // might not be correct in general.
  // TODO: Use pagesize read from the ELF aux vector instead of EXEC_PAGESIZE.

#ifdef SYS_mmap2
  offset /= EXEC_PAGESIZE;
  long syscall_number = SYS_mmap2;
#elif defined(SYS_mmap)
  long syscall_number = SYS_mmap;
#else
#error "mmap or mmap2 syscalls not available."
#endif

  // We add an explicit cast to silence a "implicit conversion loses integer
  // precision" warning when compiling for 32-bit systems.
  long mmap_offset = static_cast<long>(offset);
  long ret =
      LIBC_NAMESPACE::syscall_impl(syscall_number, reinterpret_cast<long>(addr),
                                   size, prot, flags, fd, mmap_offset);

  // The mmap/mmap2 syscalls return negative values on error. These negative
  // values are actually the negative values of the error codes. So, fix them
  // up in case an error code is detected.
  //
  // A point to keep in mind for the fix up is that a negative return value
  // from the syscall can also be an error-free value returned by the syscall.
  // However, since a valid return address cannot be within the last page, a
  // return value corresponding to a location in the last page is an error
  // value.
  if (ret < 0 && ret > -EXEC_PAGESIZE) {
    libc_errno = static_cast<int>(-ret);
    return MAP_FAILED;
  }

  return reinterpret_cast<void *>(ret);
}

} // namespace LIBC_NAMESPACE_DECL
