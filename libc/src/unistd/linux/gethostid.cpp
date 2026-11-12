//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of gethostid.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/gethostid.h"

#include "hdr/fcntl_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/close.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/open.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/read.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long, gethostid, (void)) {
  auto fd = linux_syscalls::open("/etc/hostid", O_RDONLY | O_CLOEXEC, 0);
  if (!fd)
    return 0L;

  uint32_t id = 0;
  auto ret = linux_syscalls::read(*fd, &id, sizeof(id));
  linux_syscalls::close(*fd);
  if (!ret || *ret != sizeof(id))
    return 0L;

  return static_cast<long>(id);
}

} // namespace LIBC_NAMESPACE_DECL
