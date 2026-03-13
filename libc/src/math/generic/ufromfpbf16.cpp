//===-- Implementation of ufromfpbf16 function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ufromfpbf16.h"
#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, ufromfpbf16,
                   (bfloat16 x, int rnd, unsigned int width)) {
  return fputil::fromfp</*IsSigned=*/false>(x, rnd, width);
}

} // namespace LIBC_NAMESPACE_DECL
