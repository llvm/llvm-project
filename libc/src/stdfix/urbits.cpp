//===-- Implementation of urbits function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "urbits.h"

#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned fract, urbits, (uint_ur_t x)) {
  return fixed_point::fxbits<unsigned fract, uint_ur_t>(x);
}

} // namespace LIBC_NAMESPACE_DECL
