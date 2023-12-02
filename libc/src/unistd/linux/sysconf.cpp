//===-- Linux implementation of sysconf -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/sysconf.h"

#include "config/linux/app.h"
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"
#include "src/sys/auxv/getauxval.h"
#include <unistd.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(long, sysconf, (int name)) {
  long ret = 0;
  if (name == _SC_PAGESIZE) {
    if (app.page_size)
      return app.page_size;
    int errno_backup = libc_errno;
    ret = static_cast<long>(getauxval(AT_PAGESZ));
    libc_errno = errno_backup;
  }
  // TODO: Complete the rest of the sysconf options.
  if (ret < 0) {
    libc_errno = EINVAL;
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE
