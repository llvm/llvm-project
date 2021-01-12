//===---------- Linux implementation of the POSIX mmap function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/mmap.h"

#include "config/linux/syscall.h" // For internal syscall function.
#include "include/sys/syscall.h"  // For syscall numbers.
#include "src/__support/common.h"
#include "src/errno/llvmlibc_errno.h"

#include <linux/param.h> // For EXEC_PAGESIZE.

namespace __llvm_libc {

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
#elif SYS_mmap
  long syscall_number = SYS_mmap;
#else
#error "Target platform does not have SYS_mmap or SYS_mmap2 defined"
#endif

  long ret_val =
      __llvm_libc::syscall(syscall_number, reinterpret_cast<long>(addr), size,
                           prot, flags, fd, offset);

  // The mmap/mmap2 syscalls return negative values on error. These negative
  // values are actually the negative values of the error codes. So, fix them
  // up in case an error code is detected.
  //
  // A point to keep in mind for the fix up is that a negative return value
  // from the syscall can also be an error-free value returned by the syscall.
  // However, since a valid return address cannot be within the last page, a
  // return value corresponding to a location in the last page is an error
  // value.
  if (ret_val < 0 && ret_val > -EXEC_PAGESIZE) {
    llvmlibc_errno = -ret_val;
    return MAP_FAILED;
  }

  return reinterpret_cast<void *>(ret_val);
}

} // namespace __llvm_libc
