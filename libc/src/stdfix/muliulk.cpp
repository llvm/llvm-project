//===-- Implementation of muliulk function  -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "muliulk.h"
#include "include/llvm-libc-macros/stdfix-macros.h" // unsigned long accum
#include "src/__support/common.h"                   // LLVM_LIBC_FUNCTION
#include "src/__support/fixed_point/fx_bits.h"      // fixed_point
#include "src/__support/macros/config.h"            // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned long int, muliulk, (unsigned long accum f, unsigned long int i)) {
  return fixed_point::muli(f, i);
}

} // namespace LIBC_NAMESPACE_DECL
