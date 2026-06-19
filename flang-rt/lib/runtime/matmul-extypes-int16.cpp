//===-- lib/runtime/matmul-extypes-int16.cpp --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Extended-type MATMUL instances: Int16, Unsigned16.

#include "matmul.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

// clang-format off
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
MATMUL_INSTANCE(Integer, 16, Integer, 1)
MATMUL_INSTANCE(Integer, 16, Integer, 2)
MATMUL_INSTANCE(Integer, 16, Integer, 4)
MATMUL_INSTANCE(Integer, 16, Integer, 8)
MATMUL_INSTANCE(Integer, 16, Integer, 16)
MATMUL_INSTANCE(Integer, 16, Real, 4)
MATMUL_INSTANCE(Integer, 16, Real, 8)
MATMUL_INSTANCE(Integer, 16, Complex, 4)
MATMUL_INSTANCE(Integer, 16, Complex, 8)
MATMUL_INSTANCE(Unsigned, 16, Real, 4)
MATMUL_INSTANCE(Unsigned, 16, Real, 8)
MATMUL_INSTANCE(Unsigned, 16, Complex, 4)
MATMUL_INSTANCE(Unsigned, 16, Complex, 8)
MATMUL_INSTANCE(Real, 4, Integer, 16)
MATMUL_INSTANCE(Real, 8, Integer, 16)
MATMUL_INSTANCE(Complex, 4, Integer, 16)
MATMUL_INSTANCE(Complex, 8, Integer, 16)
MATMUL_INSTANCE(Real, 4, Unsigned, 16)
MATMUL_INSTANCE(Real, 8, Unsigned, 16)
MATMUL_INSTANCE(Complex, 4, Unsigned, 16)
MATMUL_INSTANCE(Complex, 8, Unsigned, 16)
MATMUL_DIRECT_INSTANCE(Integer, 16, Integer, 1)
MATMUL_DIRECT_INSTANCE(Integer, 16, Integer, 2)
MATMUL_DIRECT_INSTANCE(Integer, 16, Integer, 4)
MATMUL_DIRECT_INSTANCE(Integer, 16, Integer, 8)
MATMUL_DIRECT_INSTANCE(Integer, 16, Integer, 16)
MATMUL_DIRECT_INSTANCE(Integer, 16, Real, 4)
MATMUL_DIRECT_INSTANCE(Integer, 16, Real, 8)
MATMUL_DIRECT_INSTANCE(Integer, 16, Complex, 4)
MATMUL_DIRECT_INSTANCE(Integer, 16, Complex, 8)
MATMUL_DIRECT_INSTANCE(Unsigned, 16, Real, 4)
MATMUL_DIRECT_INSTANCE(Unsigned, 16, Real, 8)
MATMUL_DIRECT_INSTANCE(Unsigned, 16, Complex, 4)
MATMUL_DIRECT_INSTANCE(Unsigned, 16, Complex, 8)
MATMUL_DIRECT_INSTANCE(Real, 4, Integer, 16)
MATMUL_DIRECT_INSTANCE(Real, 8, Integer, 16)
MATMUL_DIRECT_INSTANCE(Complex, 4, Integer, 16)
MATMUL_DIRECT_INSTANCE(Complex, 8, Integer, 16)
MATMUL_DIRECT_INSTANCE(Real, 4, Unsigned, 16)
MATMUL_DIRECT_INSTANCE(Real, 8, Unsigned, 16)
MATMUL_DIRECT_INSTANCE(Complex, 4, Unsigned, 16)
MATMUL_DIRECT_INSTANCE(Complex, 8, Unsigned, 16)

#if HAS_FLOAT80
MATMUL_INSTANCE(Integer, 16, Real, 10)
MATMUL_INSTANCE(Integer, 16, Complex, 10)
MATMUL_INSTANCE(Real, 10, Integer, 16)
MATMUL_INSTANCE(Complex, 10, Integer, 16)
MATMUL_INSTANCE(Unsigned, 16, Real, 10)
MATMUL_INSTANCE(Unsigned, 16, Complex, 10)
MATMUL_INSTANCE(Real, 10, Unsigned, 16)
MATMUL_INSTANCE(Complex, 10, Unsigned, 16)
MATMUL_DIRECT_INSTANCE(Integer, 16, Real, 10)
MATMUL_DIRECT_INSTANCE(Integer, 16, Complex, 10)
MATMUL_DIRECT_INSTANCE(Real, 10, Integer, 16)
MATMUL_DIRECT_INSTANCE(Complex, 10, Integer, 16)
MATMUL_DIRECT_INSTANCE(Unsigned, 16, Real, 10)
MATMUL_DIRECT_INSTANCE(Unsigned, 16, Complex, 10)
MATMUL_DIRECT_INSTANCE(Real, 10, Unsigned, 16)
MATMUL_DIRECT_INSTANCE(Complex, 10, Unsigned, 16)
#endif
#if HAS_LDBL128 || HAS_FLOAT128
MATMUL_INSTANCE(Integer, 16, Real, 16)
MATMUL_INSTANCE(Integer, 16, Complex, 16)
MATMUL_INSTANCE(Real, 16, Integer, 16)
MATMUL_INSTANCE(Complex, 16, Integer, 16)
MATMUL_INSTANCE(Unsigned, 16, Real, 16)
MATMUL_INSTANCE(Unsigned, 16, Complex, 16)
MATMUL_INSTANCE(Real, 16, Unsigned, 16)
MATMUL_INSTANCE(Complex, 16, Unsigned, 16)
MATMUL_DIRECT_INSTANCE(Integer, 16, Real, 16)
MATMUL_DIRECT_INSTANCE(Integer, 16, Complex, 16)
MATMUL_DIRECT_INSTANCE(Real, 16, Integer, 16)
MATMUL_DIRECT_INSTANCE(Complex, 16, Integer, 16)
MATMUL_DIRECT_INSTANCE(Unsigned, 16, Real, 16)
MATMUL_DIRECT_INSTANCE(Unsigned, 16, Complex, 16)
MATMUL_DIRECT_INSTANCE(Real, 16, Unsigned, 16)
MATMUL_DIRECT_INSTANCE(Complex, 16, Unsigned, 16)
#endif
#endif // __SIZEOF_INT128__
// clang-format on

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
