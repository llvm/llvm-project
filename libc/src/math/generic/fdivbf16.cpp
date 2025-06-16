//===-- Implementation of fdivbf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fdivbf16.h"

#include "src/__support/FPUtil/bfloat16.h"    // bfloat16
#include "src/__support/FPUtil/generic/div.h" // fputil::generic::mul
#include "src/__support/macros/config.h"      // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

// FIXME: 8UL % WORD_SIZE != 0, libc/src/__support/big_int.h:353:29
LLVM_LIBC_FUNCTION(bfloat16, fdivbf16, (bfloat16 x, bfloat16 y)) {
  return fputil::generic::div<bfloat16>(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
