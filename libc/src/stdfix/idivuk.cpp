//===-- Implementation of idivuk function  --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "idivuk.h"
#include "include/llvm-libc-macros/stdfix-macros.h" // unsigned accum
#include "src/__support/common.h"                   // LLVM_LIBC_FUNCTION
#include "src/__support/fixed_point/fx_bits.h"      // fixed_point
#include "src/__support/macros/config.h"            // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned int, idivuk, (unsigned accum x, unsigned accum y)) {
  return fixed_point::idiv<unsigned accum, unsigned int>(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
