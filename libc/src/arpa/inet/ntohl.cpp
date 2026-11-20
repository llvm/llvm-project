//===-- Implementation of ntohl function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/arpa/inet/ntohl.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/net/byte_order.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(uint32_t, ntohl, (uint32_t netlong)) {
  return net::ntoh(netlong);
}

} // namespace LIBC_NAMESPACE_DECL
