//===-- Implementation of the lgamma_r function for GPU -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/lgamma_r.h"
#include "src/__support/common.h"

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

extern "C" double __attribute__((overloadable)) __spirv_ocl_lgamma_r(double, int*);
LLVM_LIBC_FUNCTION(double, lgamma_r, (double x, int *signp)) {
  return __spirv_ocl_lgamma_r(x, signp);
}

} // namespace LIBC_NAMESPACE_DECL
