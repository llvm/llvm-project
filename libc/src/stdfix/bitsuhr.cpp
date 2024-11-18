//===-- Implementation of bitsuhr function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bitsuhr.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(uint_uhr_t, bitsuhr, (unsigned short fract x)) {
  return cpp::bit_cast<uint_uhr_t, unsigned short fract>(x);
}

} // namespace LIBC_NAMESPACE_DECL
