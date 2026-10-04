//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation for inet_pton
///
//===----------------------------------------------------------------------===//

#include "inet_pton.h"

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/sys_socket_macros.h"
#include "hdr/types/struct_in6_addr.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/net/address.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, inet_pton,
                   (int af, const char *__restrict src, void *__restrict dst)) {
  LIBC_CRASH_ON_NULLPTR(src);
  LIBC_CRASH_ON_NULLPTR(dst);
  if (af != AF_INET) {
    libc_errno = EAFNOSUPPORT;
    return -1;
  }
  return net::str_to_ipv4(src, *reinterpret_cast<struct in_addr *>(dst));
}

} // namespace LIBC_NAMESPACE_DECL
