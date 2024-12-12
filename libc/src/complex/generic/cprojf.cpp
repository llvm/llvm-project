//===-- Implementation of cprojf function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/complex/cprojf.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/complex_type.h"
#include "src/__support/FPUtil/BasicOperations.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(_Complex float, cprojf, (_Complex float x)) {
  Complex<float> x_c = cpp::bit_cast<Complex<float>>(x);
  if(fputil::FPBits<float>(x_c.real).is_inf() || fputil::FPBits<float>(x_c.imag).is_inf()) {
    return cpp::bit_cast<_Complex float>(Complex<float>{(fputil::FPBits<float>::inf(Sign::POS).get_val()), (float)(x_c.imag > 0 ? 0.0 : -0.0)});
  } else {
    return x;
  }
}

} // namespace LIBC_NAMESPACE_DECL
