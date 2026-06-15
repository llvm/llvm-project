//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of the in6addr_loopback constant.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_NETINET_IN6ADDR_LOOPBACK_H
#define LLVM_LIBC_SRC_NETINET_IN6ADDR_LOOPBACK_H

#include "hdr/types/struct_in6_addr.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

extern const struct in6_addr in6addr_loopback;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_NETINET_IN6ADDR_LOOPBACK_H
