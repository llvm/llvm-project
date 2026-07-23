//===-- Implementation of inet_addr function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/arpa/inet/inet_addr.h"
#include "hdr/netinet_in_macros.h"
#include "hdr/types/in_addr_t.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/common.h"
#include "src/__support/net/address.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(in_addr_t, inet_addr, (const char *cp)) {
  cpp::optional<in_addr_t> addr = net::inet_addr(cp);
  if (!addr.has_value())
    return INADDR_NONE;
  return addr.value();
}

} // namespace LIBC_NAMESPACE_DECL
