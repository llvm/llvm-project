//===-- Linux implementation of realpath -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/realpath.h"

#include "hdr/fcntl_macros.h"
#include "hdr/limits_macros.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/fcntl/open.h"
#include "src/string/strcpy.h"
#include "src/string/strdup.h"
#include "src/sys/stat/fstat.h"
#include "src/sys/stat/stat.h"
#include "src/unistd/close.h"
#include "src/unistd/readlink.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

char *set_errno_and_return(int err) {
  libc_errno = err;
  return nullptr;
}

void append_fd(char *fd_path, int fd) {
  char digits[16];
  int count = 0;
  unsigned int value = static_cast<unsigned int>(fd);
  do {
    digits[count++] = static_cast<char>('0' + value % 10);
    value /= 10;
  } while (value != 0);

  char *out = fd_path;
  constexpr char prefix[] = "/proc/self/fd/";
  for (char c : prefix) {
    if (c == '\0')
      break;
    *out++ = c;
  }
  while (count > 0)
    *out++ = digits[--count];
  *out = '\0';
}

} // namespace

LLVM_LIBC_FUNCTION(char *, realpath,
                   (const char *path, char *resolved_path)) {
  if (path == nullptr)
    return set_errno_and_return(EINVAL);

  int fd = LIBC_NAMESPACE::open(path, O_PATH | O_CLOEXEC);
  if (fd == -1)
    return nullptr;

  struct stat st;
  if (LIBC_NAMESPACE::fstat(fd, &st) == -1) {
    LIBC_NAMESPACE::close(fd);
    return nullptr;
  }
  dev_t st_dev = st.st_dev;
  ino_t st_ino = st.st_ino;

  char fd_path[sizeof("/proc/self/fd/") + 16];
  append_fd(fd_path, fd);

  char dst[PATH_MAX];
  ssize_t len = LIBC_NAMESPACE::readlink(fd_path, dst, sizeof(dst) - 1);
  if (len == -1) {
    LIBC_NAMESPACE::close(fd);
    return nullptr;
  }
  dst[len] = '\0';

  if (LIBC_NAMESPACE::stat(dst, &st) == -1 || st_dev != st.st_dev ||
      st_ino != st.st_ino) {
    LIBC_NAMESPACE::close(fd);
    return set_errno_and_return(ENOENT);
  }

  LIBC_NAMESPACE::close(fd);
  return resolved_path != nullptr ? LIBC_NAMESPACE::strcpy(resolved_path, dst)
                                  : LIBC_NAMESPACE::strdup(dst);
}

} // namespace LIBC_NAMESPACE_DECL
