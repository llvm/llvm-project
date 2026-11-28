//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of gethostname.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/gethostname.h"

#include "hdr/types/size_t.h"
#include "hdr/types/struct_utsname.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/uname.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, gethostname, (char *name, size_t size)) {
  // Check for invalid pointer
  if (name == nullptr) {
    libc_errno = EFAULT;
    return -1;
  }

  // Because there is no SYS_gethostname syscall, we use uname to get the
  // hostname.
  utsname unameData;
  auto result = linux_syscalls::uname(&unameData);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }

  // Copy up to size - 1 bytes and NUL-terminate when size > 0.
  size_t len = internal::strlcpy({name, size}, unameData.nodename);

  // Checks if the length of the hostname was greater than or equal to size
  if (len >= size) {
    libc_errno = ENAMETOOLONG;
    return -1;
  }

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
