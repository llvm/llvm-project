//===-- Implementation of fabsbf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fabsbf16.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/math/fabsbf16.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::BFloat16;

LLVM_LIBC_FUNCTION(bfloat16, fabsbf16, (bfloat16 x)) {
  return cpp::bit_cast<bfloat16>(math::fabsbf16(cpp::bit_cast<BFloat16>(x)));
}

} // namespace LIBC_NAMESPACE_DECL
