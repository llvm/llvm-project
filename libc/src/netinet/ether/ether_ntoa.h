//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header of ether_ntoa.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_NETINET_ETHER_ETHER_NTOA_H
#define LLVM_LIBC_SRC_NETINET_ETHER_ETHER_NTOA_H

#include "hdr/types/struct_ether_addr.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

char *ether_ntoa(const struct ether_addr *addr);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_NETINET_ETHER_ETHER_NTOA_H
