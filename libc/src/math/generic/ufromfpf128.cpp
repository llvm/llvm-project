//===-- Implementation of ufromfpf128 function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ufromfpf128.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/math/ufromfpf128.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::Float128;

LLVM_LIBC_FUNCTION(float128, ufromfpf128,
                   (float128 x, int rnd, unsigned int width)) {
  return cpp::bit_cast<float128>(
      math::ufromfpf128(cpp::bit_cast<Float128>(x), rnd, width));
}

} // namespace LIBC_NAMESPACE_DECL
