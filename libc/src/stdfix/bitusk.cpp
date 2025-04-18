//===-- Implementation for bitsuk function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bitsuk.h"
#include "include/llvm-libc-macros/stdfix-macros.h" // unsigned accum
#include "include/llvm-libc-types/stdfix-types.h"   // uint_uk_t
#include "src/__support/common.h"                   // LLVM_LIBC_FUNCTION
#include "src/__support/fixed_point/fx_bits.h"      // fixed_point
#include "src/__support/macros/config.h"            // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(uint_uk_t, bitsuk, (unsigned accum f)) {
  return fixed_point::bitsfx<unsigned accum, uint_uk_t>(f);
}

} // namespace LIBC_NAMESPACE_DECL
