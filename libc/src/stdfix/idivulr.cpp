//===-- Implementation of idivulr function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "idivulr.h"
#include "include/llvm-libc-macros/stdfix-macros.h" // unsigned long fract
#include "src/__support/common.h"                   // LLVM_LIBC_FUNCTION
#include "src/__support/fixed_point/fx_bits.h"      // fixed_point
#include "src/__support/macros/config.h"            // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned long int, idivulr,
                   (unsigned long fract x, unsigned long fract y)) {
  return fixed_point::idiv<unsigned long fract, unsigned long int>(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
