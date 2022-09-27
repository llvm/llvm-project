//===-- Linux implementation of stat --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/stat/stat.h"
#include "kernel_statx.h"

#include "src/__support/common.h"

#include <fcntl.h>
#include <sys/stat.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, stat,
                   (const char *__restrict path,
                    struct stat *__restrict statbuf)) {
  return statx(AT_FDCWD, path, 0, statbuf);
}

} // namespace __llvm_libc
