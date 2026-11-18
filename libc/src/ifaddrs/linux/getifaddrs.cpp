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
#include "hdr/errno_macros.h"
#include "hdr/types/struct_ifaddrs.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getifaddrs, ([[maybe_unused]] struct ifaddrs * *ifap)) {
  // TODO: Implement getifaddrs.
  libc_errno = ENOSYS;
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
