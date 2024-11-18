//===-- Implementation of bitshk function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bitshk.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(int_hk_t, bitshk, (short accum x)) {
  return cpp::bit_cast<int_hk_t, short accum>(x);
}

} // namespace LIBC_NAMESPACE_DECL
