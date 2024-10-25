//===-- Implementation of cimagf128 function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/complex/cimagf128.h"
#if defined(LIBC_TYPES_HAS_CFLOAT128)

#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float128, cimagf128, (cfloat128 x)) {
  float128 *xCmplxPtr = reinterpret_cast<float128 *>(&x);
  return xCmplxPtr[1];
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_HAS_CFLOAT128
