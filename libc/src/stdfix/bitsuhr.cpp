//===-- Implementation of bitsuhr function  -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bitsuhr.h"
#include "include/llvm-libc-macros/stdfix-macros.h" // unsigned short fract
#include "include/llvm-libc-types/uint_uhr_t.h"
#include "src/__support/common.h"                   // LLVM_LIBC_FUNCTION
#include "src/__support/fixed_point/fx_bits.h"      // fixed_point
#include "src/__support/macros/config.h"            // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(uint_uhr_t, bitsuhr, (unsigned short fract f)) {
  return fixed_point::bitsfx<unsigned short fract, uint_uhr_t>(f);
}

} // namespace LIBC_NAMESPACE_DECL
