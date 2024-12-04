//===-- Implementation of conjf function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/complex/conjf.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/complex_type.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(_Complex float, conjf, (_Complex float x)) {
  Complex<float> x_c = cpp::bit_cast<Complex<float>>(x);
  return (x_c.real - x_c.imag * (_Complex float)1.0i);
}

} // namespace LIBC_NAMESPACE_DECL
