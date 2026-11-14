//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of ether_aton_r function.
///
//===----------------------------------------------------------------------===//

#include "src/netinet/ether/ether_aton_r.h"
#include "hdr/types/struct_ether_addr.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/netinet/ether/ether_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(struct ether_addr *, ether_aton_r,
                   (const char *asc, struct ether_addr *addr)) {
  LIBC_CRASH_ON_NULLPTR(asc);
  LIBC_CRASH_ON_NULLPTR(addr);
  cpp::optional<struct ether_addr> result = net::ether_aton(asc);
  if (!result)
    return nullptr;
  *addr = *result;
  return addr;
}

} // namespace LIBC_NAMESPACE_DECL
