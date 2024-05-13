//===-- Linux implementation of fpathconf ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/errno/libc_errno.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

static long filesizebits(const struct statfs &s) {
  switch (s.f_type) {
  case JFFS2_SUPER_MAGIC:
  case MSDOS_SUPER_MAGIC:
  case NCP_SUPER_MAGIC:
    return 32;
  }
  return 64;
}

static long link_max(const struct statfs &s) {
  switch (s.f_type) {
  case EXT2_SUPER_MAGIC:
    return 32000;
  case MINIX_SUPER_MAGIC:
    return 250;
  case MINIX2_SUPER_MAGIC:
    return 65530;
  case REISERFS_SUPER_MAGIC:
    return 0xffff - 1000;
  case UFS_MAGIC:
    return 32000;
  }
  return LINK_MAX;
}

static long _2_symlinks(const struct statfs &s) {
  switch (s.f_type) {
  case ADFS_SUPER_MAGIC:
  case BFS_MAGIC:
  case CRAMFS_MAGIC:
  case EFS_SUPER_MAGIC:
  case MSDOS_SUPER_MAGIC:
  case QNX4_SUPER_MAGIC:
    return 0;
  }
  return 1;
}

static long fpathconfig(const struct fstatfs &s, int name) {
  switch (name) {
  case _PC_LINK_MAX:
    return link_max(s);

  case _PC_FILESIZEBITS:
    return filesizebits(s);

  case _PC_2_SYMLINKS:
    return _2_symlinks(s);

  case _PC_REC_MIN_XFER_SIZE:
    return s.f_bsize;

  case _PC_ALLOC_SIZE_MIN:
  case _PC_REC_XFER_ALIGN:
    return s.f_frsize;

  case _PC_MAX_CANON:
    return _POSIX_MAX_CANON;

  case _PC_MAX_INPUT:
    return _POSIX_MAX_INPUT;

  case _PC_NAME_MAX:
    return s.f_namemax;

  case _PC_PATH_MAX:
    return _POSIX_PATH_MAX;

  case _PC_PIPE_BUF:
    return _POSIX_PIPE_BUF;

  case _PC_CHOWN_RESTRICTED:
    return _POSIX_CHOWN_RESTRICTED;

  case _PC_NO_TRUNC:
    return _POSIX_NO_TRUNC;

  case _PC_VDISABLE:
    return _POSIX_VDISABLE;

  case _PC_ASYNC_IO:
  case _PC_PRIO_IO:
  case _PC_REC_INCR_XFER_SIZE:
  case _PC_REC_MAX_XFER_SIZE:
  case _PC_SYMLINK_MAX:
  case _PC_SYNC_IO:
    return -1;

  default:
    errno = EINVAL;
    return -1;
  }
}

LLVM_LIBC_FUNCTION(long, fpathconf, (int fd, int name)) {
  struct fstatfs sb;
  if (cpp::optional<LinuxStatFs> result = linux_fstatfs(fd))
    return fpathconfig(sb, name);
  return -1;
}

} // namespace LIBC_NAMESPACE
