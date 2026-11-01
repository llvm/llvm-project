//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of killpg.
///
//===----------------------------------------------------------------------===//

#include "src/signal/killpg.h"

#include "hdr/errno_macros.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, killpg, (pid_t pgrp, int sig)) {
  if (pgrp < 0) {
    libc_errno = EINVAL;
    return -1;
  }

  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_kill, -pgrp, sig);
  if (ret != 0) {
    libc_errno = (ret > 0 ? ret : -ret);
    return -1;
  }

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
