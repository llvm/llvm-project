//===-- Implementation of idivlk function  --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "idivlk.h"
#include "include/llvm-libc-macros/stdfix-macros.h" // long accum
#include "src/__support/common.h"                   // LLVM_LIBC_FUNCTION
#include "src/__support/fixed_point/fx_bits.h"      // fixed_point
#include "src/__support/macros/config.h"            // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long int, idivlk, (long accum x, long accum y)) {
  return fixed_point::idiv<long accum, long int>(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
