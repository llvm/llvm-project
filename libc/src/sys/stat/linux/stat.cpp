//===-- Linux implementation of stat --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/stat/stat.h"
#include "hdr/fcntl_macros.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/OSUtil/linux/stat/stat_via_statx.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, stat,
                   (const char *__restrict path,
                    struct stat *__restrict statbuf)) {
  ErrorOr<int> result = internal::stat_via_statx(AT_FDCWD, path, 0, statbuf);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
