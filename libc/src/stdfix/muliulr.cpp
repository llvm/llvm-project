//===-- Implementation of muliulr function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "muliulr.h"
#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned long int, muliulr,
                   (unsigned long int n, unsigned long fract m)) {
  return fixed_point::mulifx<unsigned long int, unsigned long fract>(n, m);
}

} // namespace LIBC_NAMESPACE_DECL
