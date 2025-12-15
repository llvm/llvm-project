//===-- Implementation header for expf --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_STATIC_ROUNDING_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_STATIC_ROUNDING_H

#include "expf.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

// Remark: "#pragma STDC FENV_ACCESS" might generate unsupported warnings on
// certain platforms.
#pragma STDC FENV_ACCESS ON

// Directional rounding version of expf.
LIBC_INLINE static float expf(float x, int rounding_mode) {
  int current_rounding_mode = fputil::get_round();
  if (rounding_mode == current_rounding_mode)
    return expf(x);

  fputil::set_round(rounding_mode);
  float result = expf(x);
  fputil::set_round(current_rounding_mode);
  return result;
}

#pragma STDC FENV_ACCESS DEFAULT

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_STATIC_ROUNDING_H
