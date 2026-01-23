//===-- Implementation of scalblnbf16 function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/scalblnbf16.h"
#include "hdr/float_macros.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#if FLT_RADIX != 2
#error "FLT_RADIX != 2 is not supported."
#endif

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, scalblnbf16, (bfloat16 x, long n)) {
  return fputil::ldexp(x, n);
}

} // namespace LIBC_NAMESPACE_DECL
