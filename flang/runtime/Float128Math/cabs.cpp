//===-- runtime/Float128Math/cabs.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "math-entries.h"

namespace Fortran::runtime {
extern "C" {
#if 0
// FIXME: temporarily disabled. Need to add pure C entry point
// using C _Complex ABI.
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
// NOTE: Flang calls the runtime APIs using C _Complex ABI
CppTypeFor<TypeCategory::Real, 16> RTDEF(CAbsF128)(CFloat128ComplexType x) {
  return CAbs<true>::invoke(x);
}
#endif
#endif
} // extern "C"
} // namespace Fortran::runtime
