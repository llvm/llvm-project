//===-- Implementation of inet_aton function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/arpa/inet/inet_aton.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/common.h"
#include "src/__support/net/address.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, inet_aton, (const char *cp, struct in_addr *inp)) {
  cpp::optional<in_addr_t> addr = net::inet_addr(cp);
  if (!addr.has_value())
    return 0;
  if (inp)
    inp->s_addr = addr.value();
  return 1;
}

} // namespace LIBC_NAMESPACE_DECL
