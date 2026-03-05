//===-- Linux implementation of ftok --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/sys/ipc/ftok.h"

#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"

#include "kernel_statx.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(key_t, ftok, (const char *path, int id)) {
  struct statx xbuf;

  ErrorOr<int> err = statx_for_ftok(path, xbuf);

  if (!err.has_value()) {
    libc_errno = err.error();
    return -1;
  }

  // key layout based on user input and file stats metadata
  // 31            24              16             0
  // +-------------+---------------+--------------+
  // user input id + minor dev num + file inode num
  return static_cast<key_t>(
      ((id & 0xff) << 24) |
      ((static_cast<int>(xbuf.stx_dev_minor) & 0xff) << 16) |
      (static_cast<int>(xbuf.stx_ino) & 0xffff));
}

} // namespace LIBC_NAMESPACE_DECL
