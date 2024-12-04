//===-- Implementation of conj function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/complex/conj.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/complex_type.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(_Complex double, conj, (_Complex double x)) {
  Complex<double> x_c = cpp::bit_cast<Complex<double>>(x);
  return (x_c.real - x_c.imag * 1.0i);
}

} // namespace LIBC_NAMESPACE_DECL
