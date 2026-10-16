//===-- Implementation of bitshk function  --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bitshk.h"
#include "include/llvm-libc-macros/stdfix-macros.h" // short accum
#include "include/llvm-libc-types/int_hk_t.h"
#include "src/__support/common.h"                   // LLVM_LIBC_FUNCTION
#include "src/__support/fixed_point/fx_bits.h"      // fixed_point
#include "src/__support/macros/config.h"            // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int_hk_t, bitshk, (short accum f)) {
  return fixed_point::bitsfx<short accum, int_hk_t>(f);
}

} // namespace LIBC_NAMESPACE_DECL
