//===-- Implementation of ulrbits function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ulrbits.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(unsigned long fract, ulrbits, (uint_ulr_t x)) {
  return cpp::bit_cast<unsigned long fract, uint_ulr_t>(x);
}

} // namespace LIBC_NAMESPACE_DECL
