//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the in6addr_any constant.
///
//===----------------------------------------------------------------------===//

#include "src/netinet/in6addr_any.h"
#include "hdr/netinet_in_macros.h"
#include "hdr/types/struct_in6_addr.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_VARIABLE(const struct in6_addr, in6addr_any) = IN6ADDR_ANY_INIT;

} // namespace LIBC_NAMESPACE_DECL
