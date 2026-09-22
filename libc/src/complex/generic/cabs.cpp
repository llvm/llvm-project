//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of cabs.
///
//===----------------------------------------------------------------------===//

#include "src/complex/cabs.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/complex_type.h"
#include "src/__support/macros/config.h"
#include "src/__support/math/hypot.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, cabs, (_Complex double x)) {
  Complex<double> x_c = cpp::bit_cast<Complex<double>>(x);
  return math::hypot(x_c.imag, x_c.real);
}

} // namespace LIBC_NAMESPACE_DECL
