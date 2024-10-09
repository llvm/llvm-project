//===---------- Linux implementation of the POSIX mremap function
//-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/mremap.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include <linux/param.h> // For EXEC_PAGESIZE.
#include <stdarg.h>
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, mremap,
                   (void *old_address, size_t old_size, size_t new_size,
                    int flags, ... /* void *new_address */)) {

#ifdef SYS_mremap
  long syscall_number = SYS_mremap;
#else
#error "mremap syscalls not available."
#endif
  long ret = 0;
  if (flags & MREMAP_FIXED) {
    void *arg;
    va_list varargs;
    va_start(varargs, flags);
    arg = va_arg(varargs, void *);
    va_end(varargs);
    ret = LIBC_NAMESPACE::syscall_impl(syscall_number, old_address, old_size,
                                       new_size, flags, arg);
  } else {
    ret = LIBC_NAMESPACE::syscall_impl(syscall_number, old_address, old_size,
                                       new_size, flags);
  }
  if (ret < 0 && ret > -EXEC_PAGESIZE) {
    libc_errno = static_cast<int>(-ret);
    return MAP_FAILED;
  }

  return reinterpret_cast<void *>(ret);
}

} // namespace LIBC_NAMESPACE_DECL
