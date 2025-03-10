//===-- Implementation for bitslk function  -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bitslk.h"
#include "include/llvm-libc-macros/stdfix-macros.h" // long accum
#include "include/llvm-libc-types/stdfix-types.h"   // int_lk_t
#include "src/__support/common.h"                   // LLVM_LIBC_FUNCTION
#include "src/__support/fixed_point/fx_bits.h"      // fixed_point
#include "src/__support/macros/config.h"            // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int_lk_t, bitslk, (long accum f)) {
  return fixed_point::bitsfx<long accum, int_lk_t>(f);
}

} // namespace LIBC_NAMESPACE_DECL
