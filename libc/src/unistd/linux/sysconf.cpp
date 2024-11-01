//===-- Linux implementation of sysconf -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/sysconf.h"

#include "src/__support/common.h"

#include "src/errno/libc_errno.h"
#include <linux/param.h> // For EXEC_PAGESIZE.
#include <unistd.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(long, sysconf, (int name)) {
  long ret = 0;
  if (name == _SC_PAGESIZE) {
    // TODO: get this information from the auxvector.
    return EXEC_PAGESIZE;
  }
  // TODO: Complete the rest of the sysconf options.
  if (ret < 0) {
    libc_errno = EINVAL;
    return -1;
  }
  return ret;
}

} // namespace __llvm_libc
