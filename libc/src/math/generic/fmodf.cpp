//===-- Single-precision fmodf function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmodf.h"
#include "src/__support/FPUtil/generic/FMod.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, fmodf, (float x, float y)) {
  return fputil::generic::FMod<float>::eval(x, y);
}

} // namespace __llvm_libc
