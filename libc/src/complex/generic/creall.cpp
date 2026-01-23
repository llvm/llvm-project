//===-- Implementation of creall function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/complex/creall.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/complex_type.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, creall, (_Complex long double x)) {
  Complex<long double> x_c = cpp::bit_cast<Complex<long double>>(x);
  return x_c.real;
}

} // namespace LIBC_NAMESPACE_DECL
