//===-- Implementation of lkdivi function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lkdivi.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long accum, lkdivi, (long int n, long int d)) {
  return fixed_point::fxdivi<long accum, long int>(n, d);
}

} // namespace LIBC_NAMESPACE_DECL
