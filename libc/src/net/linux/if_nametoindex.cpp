//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of if_nametoindex.
///
//===----------------------------------------------------------------------===//

#include "src/net/if_nametoindex.h"
#include "hdr/errno_macros.h"
#include "hdr/net_if_macros.h"
#include "hdr/sys_ioctl_macros.h"
#include "hdr/sys_socket_macros.h"
#include "hdr/types/struct_ifreq.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/close.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/ioctl.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/socket.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/null_check.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned int, if_nametoindex, (const char *ifname)) {
  LIBC_CRASH_ON_NULLPTR(ifname);

  cpp::string_view name(ifname);
  if (name.empty() || name.size() >= IF_NAMESIZE) {
    libc_errno = ENODEV;
    return 0;
  }

  ErrorOr<int> fd =
      linux_syscalls::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  if (!fd.has_value()) {
    libc_errno = fd.error();
    return 0;
  }

  struct ifreq ifr;
  inline_memcpy(ifr.ifr_name, name.data(), name.size() + 1);

  ErrorOr<int> ioctl_res = linux_syscalls::ioctl(*fd, SIOCGIFINDEX, &ifr);

  if (ErrorOr<int> res = linux_syscalls::close(*fd); !res.has_value()) {
    libc_errno = res.error();
    return 0;
  }

  if (!ioctl_res.has_value()) {
    libc_errno = ioctl_res.error();
    return 0;
  }

  return static_cast<unsigned int>(ifr.ifr_ifindex);
}

} // namespace LIBC_NAMESPACE_DECL
