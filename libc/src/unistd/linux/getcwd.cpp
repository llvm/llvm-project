//===-- Linux implementation of getcwd ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getcwd.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/string/string_utils.h" // For strdup.

#include <errno.h>
#include <linux/limits.h> // This is safe to include without any name pollution.
#include <stdlib.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

namespace {

bool getcwd_syscall(char *buf, size_t size) {
  int ret = __llvm_libc::syscall_impl(SYS_getcwd, buf, size);
  if (ret < 0) {
    errno = -ret;
    return false;
  } else if (ret == 0 || buf[0] != '/') {
    errno = ENOENT;
    return false;
  }
  return true;
}

} // anonymous namespace

LLVM_LIBC_FUNCTION(char *, getcwd, (char *buf, size_t size)) {
  if (buf == nullptr) {
    // We match glibc's behavior here and return the cwd in a malloc-ed buffer.
    // We will allocate a static buffer of size PATH_MAX first and fetch the cwd
    // into it. This way, if the syscall fails, we avoid unnecessary malloc
    // and free.
    char pathbuf[PATH_MAX];
    if (!getcwd_syscall(pathbuf, PATH_MAX))
      return nullptr;
    char *cwd = internal::strdup(pathbuf);
    if (cwd == nullptr) {
      errno = ENOMEM;
      return nullptr;
    }
    return cwd;
  } else if (size == 0) {
    errno = EINVAL;
    return nullptr;
  }

  // TODO: When buf is not sufficient, evaluate the full cwd path using
  // alternate approaches.

  if (!getcwd_syscall(buf, size))
    return nullptr;
  return buf;
}

} // namespace __llvm_libc
