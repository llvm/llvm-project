//===-- Implementation of urdivi function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "urdivi.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned fract, urdivi, (unsigned int n, unsigned int d)) {
  return fixed_point::fxdivi<unsigned fract, unsigned int>(n, d);
}

} // namespace LIBC_NAMESPACE_DECL
