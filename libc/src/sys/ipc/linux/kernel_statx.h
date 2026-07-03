//===-- Wrapper over SYS_statx syscall for ftok ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_IPC_LINUX_KERNEL_STATX_H
#define LLVM_LIBC_SRC_SYS_IPC_LINUX_KERNEL_STATX_H

#include "hdr/fcntl_macros.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/error_or.h"
#include "sys/syscall.h"

#include <linux/stat.h>

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE ErrorOr<int> statx_for_ftok(const char *path, struct statx &xbuf) {

  // store the file stats metadata into xbuf
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_statx, AT_FDCWD, path, 0,
                                              STATX_BASIC_STATS, &xbuf);

  if (ret < 0)
    return Error(-ret);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_IPC_LINUX_KERNEL_STATX_H
