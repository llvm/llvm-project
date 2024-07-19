//===-- Linux implementation of pathconf ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/pathconf.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include "src/sys/statvfs/linux/statfs_utils.h"
#include "src/unistd/linux/pathconf_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long, pathconf, (const char *path, int name)) {
  if (cpp::optional<statfs_utils::LinuxStatFs> result =
          statfs_utils::linux_statfs(path))
    return pathconfig(result.value(), name);
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
