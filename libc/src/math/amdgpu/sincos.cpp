//===-- Implementation of the sincos function for GPU ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sincos.h"
#include "src/__support/common.h"

#include "declarations.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, sincos, (double x, double *sinptr, double *cosptr)) {
  *sinptr = __ocml_sincos_f64(x, cosptr);
}

} // namespace LIBC_NAMESPACE_DECL
