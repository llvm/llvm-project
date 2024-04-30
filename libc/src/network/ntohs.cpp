//===-- Implementation of ntohs function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/network/ntohs.h"
#include "src/__support/common.h"
#include "src/__support/endian.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(uint16_t, ntohs, (uint16_t netshort)) {
  if constexpr (Endian::IS_LITTLE)
    return __builtin_bswap16(netshort);
  else
    return netshort;
}

} // namespace LIBC_NAMESPACE
