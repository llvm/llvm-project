//===-- Implementation of ulrdivi function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ulrdivi.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned long fract, ulrdivi,
                   (unsigned long int n, unsigned long int d)) {
  return fixed_point::fxdivi<unsigned long fract, unsigned long int>(n, d);
}

} // namespace LIBC_NAMESPACE_DECL
