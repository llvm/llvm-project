//===-- Implementation of hypotbf16 function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/hypotbf16.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/math/hypotbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(__bf16, hypotbf16, (__bf16 x, __bf16 y)) {
  return cpp::bit_cast<__bf16>(
      math::hypotbf16(cpp::bit_cast<bfloat16>(x), cpp::bit_cast<bfloat16>(y)));
}

} // namespace LIBC_NAMESPACE_DECL
