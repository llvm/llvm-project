//===-- Implementation of htonl function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/network/htonl.h"
#include "src/__support/common.h"
#include "src/__support/endian.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(uint32_t, htonl, (uint32_t hostlong)) {
  return Endian::to_big_endian(hostlong);
}

} // namespace __llvm_libc
