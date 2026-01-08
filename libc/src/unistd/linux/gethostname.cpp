//===-- Linux implementation of gethostname -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/gethostname.h"

#include "hdr/types/size_t.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/string/string_utils.h"

#include <sys/syscall.h> // For syscall numbers.
#include <sys/utsname.h>

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
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_uname, &unameData);
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }

  // Guarantee that the name will be null terminated.
  // The amount of bytes copied is min(size + 1, strlen(nodename) + 1)
  // +1 to account for the null terminator (the last copied byte is a NULL).
  internal::strlcpy(name, unameData.nodename, size + 1);

  // Checks if the length of the hostname was greater than or equal to size
  if (internal::string_length(unameData.nodename) >= size) {
    libc_errno = ENAMETOOLONG;
    return -1;
  }

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
