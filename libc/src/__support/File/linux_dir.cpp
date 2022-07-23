//===--- Linux implementation of the Dir helpers --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dir.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.

#include <errno.h>
#include <fcntl.h>       // For open flags
#include <sys/syscall.h> // For syscall numbers

namespace __llvm_libc {

int platform_opendir(const char *name) {
  int open_flags = O_RDONLY | O_DIRECTORY | O_CLOEXEC;
#ifdef SYS_open
  int fd = __llvm_libc::syscall(SYS_open, name, open_flags);
#elif defined(SYS_openat)
  int fd = __llvm_libc::syscall(SYS_openat, AT_FDCWD, name, open_flags);
#else
#error                                                                         \
    "SYS_open and SYS_openat syscalls not available to perform an open operation."
#endif

  if (fd < 0) {
    errno = -fd;
    return -1;
  }
  return fd;
}

size_t platform_fetch_dirents(int fd, cpp::MutableArrayRef<uint8_t> buffer) {
  long size =
      __llvm_libc::syscall(SYS_getdents, fd, buffer.data(), buffer.size());
  if (size < 0) {
    errno = -size;
    return 0;
  }
  return size;
}

bool platform_closedir(int fd) {
  long ret = __llvm_libc::syscall(SYS_close, fd);
  if (ret < 0) {
    errno = -ret;
    return false;
  }
  return true;
}

} // namespace __llvm_libc
