//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of ether_ntoa function.
///
//===----------------------------------------------------------------------===//

#include "src/netinet/ether/ether_ntoa.h"
#include "hdr/types/struct_ether_addr.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/netinet/ether/ether_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, ether_ntoa, (const struct ether_addr *addr)) {
  LIBC_CRASH_ON_NULLPTR(addr);
  static LIBC_THREAD_LOCAL char buffer[net::ETHER_ADDR_STR_LEN];
  return net::ether_ntoa_r(*addr, buffer);
}

} // namespace LIBC_NAMESPACE_DECL
