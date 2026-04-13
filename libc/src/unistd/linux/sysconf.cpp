//===-- Linux implementation of sysconf -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/sysconf.h"

#include "src/__support/common.h"

#include "hdr/unistd_macros.h"
#include "src/__support/OSUtil/linux/auxv.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long, sysconf, (int name)) {
  long ret = 0;
  if (name == _SC_PAGESIZE) {
    cpp::optional<unsigned long> page_size = auxv::get(AT_PAGESZ);
    if (page_size)
      return static_cast<long>(*page_size);
    ret = -1;
  }

  // TODO: Complete the rest of the sysconf options.
  if (ret < 0) {
    libc_errno = EINVAL;
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
