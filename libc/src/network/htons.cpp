//===-- Implementation of htons function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/network/htons.h"
#include "src/__support/common.h"
#include "src/__support/endian.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(uint16_t, htons, (uint16_t hostshort)) {
  return Endian::to_big_endian(hostshort);
}

} // namespace LIBC_NAMESPACE
