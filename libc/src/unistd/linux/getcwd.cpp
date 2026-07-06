//===-- Linux implementation of getcwd ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getcwd.h"

#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getcwd.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/string/allocating_string_utils.h" // For strdup.

#include <linux/limits.h> // This is safe to include without any name pollution.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, getcwd, (char *buf, size_t size)) {
  if (buf == nullptr) {
    // We match glibc's behavior here and return the cwd in a malloc-ed buffer.
    // We will allocate a static buffer of size PATH_MAX first and fetch the cwd
    // into it. This way, if the syscall fails, we avoid unnecessary malloc
    // and free.
    char pathbuf[PATH_MAX];

    ErrorOr<ssize_t> bytes_written = linux_syscalls::getcwd(pathbuf, PATH_MAX);
    if (!bytes_written) {
      libc_errno = bytes_written.error();
      return nullptr;
    }

    cpp::optional<char *> cwd = internal::strdup(pathbuf);
    if (!cwd) {
      libc_errno = ENOMEM;
      return nullptr;
    }
    return *cwd;
  }

  if (size == 0) {
    libc_errno = EINVAL;
    return nullptr;
  }

  // TODO: When buf is not sufficient, evaluate the full cwd path using
  // alternate approaches.
  ErrorOr<ssize_t> bytes_written = linux_syscalls::getcwd(buf, size);
  if (!bytes_written) {
    libc_errno = bytes_written.error();
    return nullptr;
  }

  return buf;
}

} // namespace LIBC_NAMESPACE_DECL
