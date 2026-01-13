//===-- Implementation of bitslr function  --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bitslr.h"
#include "include/llvm-libc-macros/stdfix-macros.h" // long fract
#include "include/llvm-libc-types/int_lr_t.h"       // int_lr_t
#include "src/__support/common.h"                   // LLVM_LIBC_FUNCTION
#include "src/__support/fixed_point/fx_bits.h"      // fixed_point
#include "src/__support/macros/config.h"            // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int_lr_t, bitslr, (long fract f)) {
  return fixed_point::bitsfx<long fract, int_lr_t>(f);
}

} // namespace LIBC_NAMESPACE_DECL
