//===-- Implementation of the lgamma_r function for GPU -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/lgamma_r.h"
#include "src/__support/common.h"

#include "declarations.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, lgamma_r, (double x, int *signp)) {
  int tmp = *signp;
  double r = __ocml_lgamma_r_f64(x, (gpu::Private<int> *)&tmp);
  *signp = tmp;
  return r;
}

} // namespace LIBC_NAMESPACE_DECL
