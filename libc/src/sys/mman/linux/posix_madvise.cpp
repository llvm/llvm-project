//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of posix_madvise.
///
//===----------------------------------------------------------------------===//

#include "src/sys/mman/posix_madvise.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/madvise.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include "hdr/sys_mman_macros.h" // For POSIX_MADV_DONTNEED.

namespace LIBC_NAMESPACE_DECL {

// This function is currently linux only. It has to be refactored suitably if
// posix_madvise is to be supported on non-linux operating systems also.
LLVM_LIBC_FUNCTION(int, posix_madvise, (void *addr, size_t size, int advice)) {
  // POSIX_MADV_DONTNEED does nothing because the default MADV_DONTNEED may
  // cause data loss, which the posix madvise does not allow.
  if (advice == POSIX_MADV_DONTNEED) {
    return 0;
  }
  auto result = linux_syscalls::madvise(addr, size, advice);
  return result ? 0 : result.error();
}

} // namespace LIBC_NAMESPACE_DECL
