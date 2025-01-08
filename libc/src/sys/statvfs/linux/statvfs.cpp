//===-- Linux implementation of statvfs -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/statvfs/statvfs.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "src/sys/statvfs/linux/statfs_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, statvfs,
                   (const char *__restrict path,
                    struct statvfs *__restrict buf)) {
  using namespace statfs_utils;
  cpp::optional<LinuxStatFs> result = linux_statfs(path);
  if (result) {
    LIBC_ASSERT(buf != nullptr);
    *buf = statfs_to_statvfs(*result);
  }
  return result ? 0 : -1;
}

} // namespace LIBC_NAMESPACE_DECL
