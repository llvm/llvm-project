//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of if_indextoname.
///
//===----------------------------------------------------------------------===//

#include "src/net/if_indextoname.h"
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
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, if_indextoname,
                   (unsigned int ifindex, char *ifname)) {
  LIBC_CRASH_ON_NULLPTR(ifname);

  ErrorOr<int> fd =
      linux_syscalls::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  if (!fd.has_value()) {
    libc_errno = fd.error();
    return nullptr;
  }

  struct ifreq ifr;
  ifr.ifr_ifindex = static_cast<int>(ifindex);

  ErrorOr<int> ioctl_res = linux_syscalls::ioctl(*fd, SIOCGIFNAME, &ifr);

  if (ErrorOr<int> res = linux_syscalls::close(*fd); !res.has_value()) {
    libc_errno = res.error();
    return nullptr;
  }

  if (!ioctl_res.has_value()) {
    // Map kernel ENODEV to POSIX-mandated ENXIO.
    libc_errno = (ioctl_res.error() == ENODEV) ? ENXIO : ioctl_res.error();
    return nullptr;
  }

  cpp::string_view k_name(
      ifr.ifr_name, internal::strnlen(ifr.ifr_name, sizeof(ifr.ifr_name)));
  inline_memcpy(ifname, k_name.data(), k_name.size());
  // Linux kernel does allow network devices exactly IF_NAMESIZE long (it leaves
  // room for the terminating \0). But if that happens, let's not overrun the
  // user provided buffer.
  if (k_name.size() < IF_NAMESIZE)
    ifname[k_name.size()] = '\0';

  return ifname;
}

} // namespace LIBC_NAMESPACE_DECL
