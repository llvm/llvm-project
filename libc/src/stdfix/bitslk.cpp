//===-- Implementation of bitslk function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bitslk.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(int_lk_t, bitslk, (long accum x)) {
  return cpp::bit_cast<int_lk_t, long accum>(x);
}

} // namespace LIBC_NAMESPACE_DECL
