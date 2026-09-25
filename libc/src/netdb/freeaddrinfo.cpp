//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of freeaddrinfo.
///
//===----------------------------------------------------------------------===//

#include "src/netdb/freeaddrinfo.h"
#include "hdr/types/struct_addrinfo.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, freeaddrinfo,
                   ([[maybe_unused]] struct addrinfo * res)) {
  // TODO: Implement freeaddrinfo.
  return;
}

} // namespace LIBC_NAMESPACE_DECL
