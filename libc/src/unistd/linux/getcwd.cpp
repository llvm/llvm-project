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
#include "src/string/allocating_string_utils.h" // For strdup.

#include "src/errno/libc_errno.h"
#include <linux/limits.h> // This is safe to include without any name pollution.
#include <stdlib.h>
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

namespace {

bool getcwd_syscall(char *buf, size_t size) {
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_getcwd, buf, size);
  if (ret < 0) {
    libc_errno = -ret;
    return false;
  } else if (ret == 0 || buf[0] != '/') {
    libc_errno = ENOENT;
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
    auto cwd = internal::strdup(pathbuf);
    if (!cwd) {
      libc_errno = ENOMEM;
      return nullptr;
    }
    return *cwd;
  } else if (size == 0) {
    libc_errno = EINVAL;
    return nullptr;
  }

  // TODO: When buf is not sufficient, evaluate the full cwd path using
  // alternate approaches.

  if (!getcwd_syscall(buf, size))
    return nullptr;
  return buf;
}

} // namespace LIBC_NAMESPACE
