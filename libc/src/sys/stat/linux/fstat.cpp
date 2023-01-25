//===-- Linux implementation of fstat -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/stat/fstat.h"
#include "kernel_statx.h"

#include "src/__support/common.h"

#include <fcntl.h>
#include <sys/stat.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, fstat, (int fd, struct stat *statbuf)) {
  return statx(fd, "", AT_EMPTY_PATH, statbuf);
}

} // namespace __llvm_libc
