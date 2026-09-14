//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of inet_ntoa function.
///
//===----------------------------------------------------------------------===//

#include "src/arpa/inet/inet_ntoa.h"
#include "hdr/inet-address-macros.h"
#include "src/__support/CPP/span.h"
#include "src/__support/common.h"
#include "src/__support/net/address.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, inet_ntoa, (struct in_addr in)) {
  static LIBC_THREAD_LOCAL char buffer[INET_ADDRSTRLEN];
  // Buffer is large enough for any address.
  (void)net::ipv4_to_str(in, cpp::span<char>(buffer, INET_ADDRSTRLEN));
  return buffer;
}

} // namespace LIBC_NAMESPACE_DECL
