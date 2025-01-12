//===-- Implementation of cimagf128 function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/complex/cimagf128.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/complex_type.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float128, cimagf128, (cfloat128 x)) {
  Complex<float128> x_c = cpp::bit_cast<Complex<float128>>(x);
  return x_c.imag;
}

} // namespace LIBC_NAMESPACE_DECL
