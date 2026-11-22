//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of shmat.
///
//===----------------------------------------------------------------------===//

#include "src/sys/shm/shmat.h"

#include "hdr/sys_shm_macros.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/shmat.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, shmat,
                   (int shmid, const void *shmaddr, int shmflg)) {
  ErrorOr<void *> ret = linux_syscalls::shmat(shmid, shmaddr, shmflg);
  if (!ret) {
    libc_errno = ret.error();
    return SHM_FAILED;
  }
  return ret.value();
}

} // namespace LIBC_NAMESPACE_DECL
