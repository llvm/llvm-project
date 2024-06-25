//===---------- Linux implementation of the POSIX mmap function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/mmap.h"

#include "config/linux/app.h"    // app
#include "hdr/sys_auxv_macros.h" // AT_PAGESZ
#include "hdr/sys_mman_macros.h" // MAP_FAILED
#include "hdr/types/off64_t.h"
#include "hdr/types/off_t.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/block.h"          // align_up
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"
#include "src/sys/auxv/getauxval.h"

#include <stddef.h>      // size_t
#include <stdint.h>      // PTRDIFF_MAX
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

// Older 32b systems generally have a SYS_mmap2 that accepts a 32b value which
// was a 64b value shifted down by 12; this magic constant isn't exposed via
// UAPI headers, but its in kernel sources for mmap2 implementations.
#ifndef __LP64__

// TODO: move these helpers to OSUtil?
#ifdef LIBC_FULL_BUILD
unsigned long get_page_size() { return app.page_size; }
#else
unsigned long get_page_size() {
  // TODO: is it ok for mmap to call getauxval in overlay mode? Or is there a
  // risk of infinite recursion?
  return ::getauxval(AT_PAGESZ);
}
#endif // LIBC_FULL_BUILD

void *mmap64(void *addr, size_t size, int prot, int flags, int fd,
             off64_t offset) {
  constexpr size_t MMAP2_SHIFT = 12;

  if (offset < 0 || offset % (1UL << MMAP2_SHIFT)) {
    libc_errno = EINVAL;
    return MAP_FAILED;
  }

  // Prevent allocations large enough for `end - start` to overflow,
  // to avoid security bugs.
  size_t rounded = align_up(size, get_page_size());
  if (rounded < size || rounded > PTRDIFF_MAX) {
    libc_errno = ENOMEM;
    return MAP_FAILED;
  }

  long ret = syscall_impl(SYS_mmap2, reinterpret_cast<long>(addr), size, prot,
                          flags, fd, static_cast<long>(offset >> MMAP2_SHIFT));

  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return MAP_FAILED;
  }

  return reinterpret_cast<void *>(ret);
}
#endif // __LP64__

// This function is currently linux only. It has to be refactored suitably if
// mmap is to be supported on non-linux operating systems also.
LLVM_LIBC_FUNCTION(void *, mmap,
                   (void *addr, size_t size, int prot, int flags, int fd,
                    off_t offset)) {
#ifdef __LP64__
  // A lot of POSIX standard prescribed validation of the parameters is not
  // done in this function as modern linux versions do it in the syscall.
  // TODO: Perform argument validation not done by the linux syscall.

  long ret = LIBC_NAMESPACE::syscall_impl(
      SYS_mmap, reinterpret_cast<long>(addr), size, prot, flags, fd, offset);

  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return MAP_FAILED;
  }

  return reinterpret_cast<void *>(ret);
#else
  return mmap64(addr, size, prot, flags, fd, static_cast<off64_t>(offset));
#endif // __LP64__
}

} // namespace LIBC_NAMESPACE
