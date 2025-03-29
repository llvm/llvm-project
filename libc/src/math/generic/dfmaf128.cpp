//===-- Implementation of dfmaf128 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_DFMAf128_H
#define LLVM_LIBC_SRC_MATH_DFMAf128_H

#include "src/math/dfmaf128.h"
#include "src/__support/FPUtil/FMA.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, dfmaf128, (float128 x, float128 y, float128 z)) {
  return fputil::fma<double>(x, y, z);
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATH_DFMAf128_H
