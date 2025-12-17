//===-- Implementation of inet_addr function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/arpa/inet/inet_addr.h"
#include "include/llvm-libc-macros/netinet-in-macros.h"
#include "include/llvm-libc-types/in_addr.h"
#include "include/llvm-libc-types/in_addr_t.h"
#include "src/__support/common.h"
#include "src/arpa/inet/inet_aton.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(in_addr_t, inet_addr, (const char *cp)) {
  in_addr addr;
  return inet_aton(cp, &addr) ? addr.s_addr : INADDR_NONE;
}

} // namespace LIBC_NAMESPACE_DECL
