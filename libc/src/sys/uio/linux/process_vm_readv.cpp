//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of process_vm_readv.
///
//===----------------------------------------------------------------------===//

#include "src/sys/uio/process_vm_readv.h"

#include "hdr/types/pid_t.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_iovec.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/process_vm_readv.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, process_vm_readv,
                   (pid_t pid, const struct iovec *local_iov,
                    unsigned long liovcnt, const struct iovec *remote_iov,
                    unsigned long riovcnt, unsigned long flags)) {
  ErrorOr<ssize_t> ret = linux_syscalls::process_vm_readv(
      pid, local_iov, liovcnt, remote_iov, riovcnt, flags);
  if (!ret) {
    libc_errno = ret.error();
    return -1;
  }
  return ret.value();
}

} // namespace LIBC_NAMESPACE_DECL
