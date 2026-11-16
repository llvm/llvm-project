//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of getifaddrs.
///
//===----------------------------------------------------------------------===//

#include "src/ifaddrs/getifaddrs.h"
#include "hdr/types/struct_ifaddrs.h"
#include "src/__support/OSUtil/linux/network_syscall_policy.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/ifaddrs/linux/getifaddrs_impl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getifaddrs, (struct ifaddrs * *ifap)) {
  LIBC_CRASH_ON_NULLPTR(ifap);
  ErrorOr<struct ifaddrs *> result =
      net::getifaddrs<net::DefaultNetworkSyscallPolicy>();
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  *ifap = *result;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
