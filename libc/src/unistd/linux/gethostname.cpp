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
#include "src/__support/macros/config.h"

#include "src/string/strlen.h"
#include "src/string/strncpy.h"
#include "src/errno/libc_errno.h"

#include <sys/syscall.h> // For syscall numbers.
#include <sys/utsname.h>

namespace LIBC_NAMESPACE_DECL {

// Matching the behavior of glibc version 2.2 and later.
// Copies up to len bytes from the returned nodename field into name.
LLVM_LIBC_FUNCTION(int, gethostname, (char *name, size_t len)) {
  
  // Check for invalid pointer
  if (name == nullptr) {
    libc_errno = EFAULT;
    return -1;
  }

  struct utsname unameData;
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_uname, &unameData);

  // Checks if the length of the nodename was greater than or equal to len, and if it is,
  // then the function returns -1 with errno set to ENAMETOOLONG.
  // In this case, a terminating null byte is not included in the returned name.
  if (strlen(unameData.nodename) >= len)
  {
    strncpy(name, unameData.nodename, len);
    libc_errno = ENAMETOOLONG;
    return -1;
  }

  // If the size of the array name is not large enough (less than the size of nodename with null termination), then anything might happen.
  // In this case, what happens to the array name will be determined by the implementation of LIBC_NAMESPACE_DECL::strncpy
  strncpy(name, unameData.nodename, len);

  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL


