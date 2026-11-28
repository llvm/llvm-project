//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// BFloat16 e^x - 1 function.
///
//===----------------------------------------------------------------------===//

#include "src/math/expm1bf16.h"
#include "src/__support/math/expm1bf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, expm1bf16, (bfloat16 x)) {
  return math::expm1bf16(x);
}

} // namespace LIBC_NAMESPACE_DECL
