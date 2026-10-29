//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of posix_fadvise.
///
//===----------------------------------------------------------------------===//

#include "src/fcntl/posix_fadvise.h"

#include "hdr/types/off_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/posix_fadvise.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, posix_fadvise,
                   (int fd, off_t offset, off_t len, int advice)) {
  auto result = linux_syscalls::posix_fadvise(fd, offset, len, advice);
  if (!result)
    return result.error();
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
