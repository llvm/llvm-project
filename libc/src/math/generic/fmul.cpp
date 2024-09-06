//===-- Implementation of fmul function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/math/fmul.h"
#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/generic/mul.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
/*
LLVM_LIBC_FUNCTION(float, fmul, (double x, double y)) {
return fputil::generic::mul<float>(x, y);
}
*/
LLVM_LIBC_FUNCTION(float, fmul, (double x, double y)) {
  fputil::DoubleDouble prod = fputil::exact_mult(x, y);
  if (LIBC_UNLIKELY(fputil::FPBits<double>(prod.hi).is_inf_or_nan() ||
                    fputil::FPBits<double>(prod.hi).is_zero()))
    return static_cast<float>(prod.hi);
  if (LIBC_UNLIKELY(fputil::FPBits<double>(prod.hi).is_inf() ||
                    fputil::FPBits<double>(prod.hi).is_zero())) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return fputil::FPBits<double>::quiet_nan().get_val();
  }
  return static_cast<float>(prod.hi + prod.lo);
}

} // namespace LIBC_NAMESPACE_DECL
