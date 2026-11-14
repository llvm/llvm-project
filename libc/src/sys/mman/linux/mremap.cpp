//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of mremap.
///
//===----------------------------------------------------------------------===//

#include "src/sys/mman/mremap.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/mremap.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <stdarg.h>

#include "hdr/sys_mman_macros.h" // For MREMAP_FIXED, MAP_FAILED.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, mremap,
                   (void *old_address, size_t old_size, size_t new_size,
                    int flags, ... /* void *new_address */)) {

  void *new_address = nullptr;
  if (flags & MREMAP_FIXED) {
    va_list varargs;
    va_start(varargs, flags);
    new_address = va_arg(varargs, void *);
    va_end(varargs);
  }
  auto result = linux_syscalls::mremap(old_address, old_size, new_size, flags,
                                       new_address);
  if (!result) {
    libc_errno = result.error();
    return MAP_FAILED;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
