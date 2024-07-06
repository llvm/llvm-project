//===-- Implementation of f16fmal function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/f16fmal.h"
#include "src/__support/FPUtil/FMA.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float16, f16fmal,
                   (long double x, long double y, long double z)) {
  return fputil::fma<float16>(x, y, z);
}

} // namespace LIBC_NAMESPACE
