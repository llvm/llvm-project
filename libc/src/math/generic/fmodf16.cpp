//===-- Implementation of fmodf16 function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmodf16.h"
#include "src/__support/FPUtil/generic/FMod.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float16, fmodf16, (float16 x, float16 y)) {
  return fputil::generic::FMod<float16, uint32_t>::eval(x, y);
}

} // namespace LIBC_NAMESPACE
