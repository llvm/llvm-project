//===-- Implementation of cimagf function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/complex/cimagf.h"
#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, cimagf, (_Complex float x)) {
  float *xCmplxPtr = reinterpret_cast<float *>(&x);
  return xCmplxPtr[1];
}

} // namespace LIBC_NAMESPACE_DECL
