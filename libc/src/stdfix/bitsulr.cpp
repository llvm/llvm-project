//===-- Implementation of bitsulr function  -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bitsulr.h"
#include "include/llvm-libc-macros/stdfix-macros.h" // unsigned long fract
#include "include/llvm-libc-types/stdfix-types.h"   // uint_ulr_t
#include "src/__support/common.h"                   // LLVM_LIBC_FUNCTION
#include "src/__support/fixed_point/fx_bits.h"      // fixed_point
#include "src/__support/macros/config.h"            // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(uint_ulr_t, bitsulr, (unsigned long fract f)) {
  return fixed_point::bitsfx<unsigned long fract, uint_ulr_t>(f);
}

} // namespace LIBC_NAMESPACE_DECL
