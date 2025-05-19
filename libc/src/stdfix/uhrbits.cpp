//===-- Implementation of uhrbits function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "uhrbits.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(unsigned short fract, uhrbits, (uint_uhr_t x)) {
  return cpp::bit_cast<unsigned short fract, uint_uhr_t>(x);
}

} // namespace LIBC_NAMESPACE_DECL
