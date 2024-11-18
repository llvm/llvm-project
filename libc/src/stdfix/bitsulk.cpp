//===-- Implementation of bitsulk function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bitsulk.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(uint_ulk_t, bitsulk, (unsigned long accum x)) {
  return cpp::bit_cast<uint_ulk_t, unsigned long accum>(x);
}

} // namespace LIBC_NAMESPACE_DECL
