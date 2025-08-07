//===-- Implementation of idivur function  --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "idivur.h"
#include "include/llvm-libc-macros/stdfix-macros.h" // unsigned fract
#include "src/__support/common.h"                   // LLVM_LIBC_FUNCTION
#include "src/__support/fixed_point/fx_bits.h"      // fixed_point
#include "src/__support/macros/config.h"            // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned int, idivur, (unsigned fract x, unsigned fract y)) {
  return fixed_point::idiv<unsigned fract, unsigned int>(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
