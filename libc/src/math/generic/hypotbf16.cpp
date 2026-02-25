//===-- Implementation of hypotbf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/math/hypotbf16.h"
#include "src/__support/math/hypotbf16.h"
#include "src/__support/FPUtil/bfloat16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(BFloat16, hypotf, (BFloat16 x, BFloat16 y)) {
  return math::hypotf(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
