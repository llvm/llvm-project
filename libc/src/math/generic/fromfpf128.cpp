//===-- Implementation of fromfpf128 function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fromfpf128.h"
#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float128, fromfpf128,
                   (float128 x, int rnd, unsigned int width)) {
  return fputil::fromfp</*IsSigned=*/true>(x, rnd, width);
}

} // namespace LIBC_NAMESPACE_DECL
