//===-- Implementation of cproj function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/complex/cproj.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/common.h"
#include "src/__support/complex_type.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(_Complex double, cproj, (_Complex double x)) {
  Complex<double> x_c = cpp::bit_cast<Complex<double>>(x);
  if (fputil::FPBits<double>(x_c.real).is_inf() ||
      fputil::FPBits<double>(x_c.imag).is_inf()) {
    return cpp::bit_cast<_Complex double>(
        Complex<double>{(fputil::FPBits<double>::inf(Sign::POS).get_val()),
                        (x_c.imag > 0 ? 0.0 : -0.0)});
  } else {
    return x;
  }
}

} // namespace LIBC_NAMESPACE_DECL
