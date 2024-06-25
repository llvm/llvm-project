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

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void, sincos, (double x, double *sinptr, double *cosptr)) {
  return __nv_sincos(x, sinptr, cosptr);
}

} // namespace LIBC_NAMESPACE
