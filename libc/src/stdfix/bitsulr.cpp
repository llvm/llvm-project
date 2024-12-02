//===-- Implementation of bitsulr function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bitsulr.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(uint_ulr_t, bitsulr, (unsigned long fract x)) {
  return cpp::bit_cast<uint_ulr_t, unsigned long fract>(x);
}

} // namespace LIBC_NAMESPACE_DECL
