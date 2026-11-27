//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of freeifaddrs.
///
//===----------------------------------------------------------------------===//

#include "src/ifaddrs/freeifaddrs.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/struct_ifaddrs.h"
#include "src/__support/CPP/new.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, freeifaddrs, (struct ifaddrs * ifa)) {
  // ifa was allocated in net::getifaddrs() as a single contiguous uint8_t[]
  // buffer holding all the list elements and the data they point to.
  delete[] reinterpret_cast<uint8_t *>(ifa);
}

} // namespace LIBC_NAMESPACE_DECL
