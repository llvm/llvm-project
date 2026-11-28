//===-- Implementation of htons function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/arpa/inet/htons.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/net/byte_order.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(uint16_t, htons, (uint16_t hostshort)) {
  return net::hton(hostshort);
}

} // namespace LIBC_NAMESPACE_DECL
