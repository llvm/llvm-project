//===-- Implementation of faddbf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/faddbf16.h"

#include "src/__support/FPUtil/bfloat16.h"    // bfloat16
#include "src/__support/FPUtil/generic/mul.h" // fputil::generic::mul
#include "src/__support/macros/config.h"      // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

// FIXME: error: shift count >= width of type [-Werror,-Wshift-count-overflow
// libc/src/__support/FPUtil/dyadic_float.h:578:78
LLVM_LIBC_FUNCTION(bfloat16, faddbf16, (bfloat16 x, bfloat16 y)) {
  return fputil::generic::mul<bfloat16>(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
