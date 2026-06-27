//===-- lib/runtime/matmul-transpose-extypes-real16.cpp ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Extended-type MATMUL-TRANSPOSE instances: Real16, Complex16.

#include "matmul-transpose.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

// clang-format off
#if HAS_LDBL128 || HAS_FLOAT128
MATMUL_INSTANCE(Integer, 1, Real, 16)       MATMUL_INSTANCE(Integer, 1, Complex, 16)
MATMUL_INSTANCE(Integer, 2, Real, 16)       MATMUL_INSTANCE(Integer, 2, Complex, 16)
MATMUL_INSTANCE(Integer, 4, Real, 16)       MATMUL_INSTANCE(Integer, 4, Complex, 16)
MATMUL_INSTANCE(Integer, 8, Real, 16)       MATMUL_INSTANCE(Integer, 8, Complex, 16)
MATMUL_INSTANCE(Real, 4, Real, 16)          MATMUL_INSTANCE(Real, 4, Complex, 16)
MATMUL_INSTANCE(Real, 8, Real, 16)          MATMUL_INSTANCE(Real, 8, Complex, 16)
MATMUL_INSTANCE(Real, 16, Integer, 1)       MATMUL_INSTANCE(Real, 16, Integer, 2)
MATMUL_INSTANCE(Real, 16, Integer, 4)       MATMUL_INSTANCE(Real, 16, Integer, 8)
MATMUL_INSTANCE(Real, 16, Real, 4)          MATMUL_INSTANCE(Real, 16, Real, 8)
MATMUL_INSTANCE(Real, 16, Real, 16)
MATMUL_INSTANCE(Real, 16, Complex, 4)       MATMUL_INSTANCE(Real, 16, Complex, 8)
MATMUL_INSTANCE(Real, 16, Complex, 16)
MATMUL_INSTANCE(Complex, 4, Real, 16)       MATMUL_INSTANCE(Complex, 4, Complex, 16)
MATMUL_INSTANCE(Complex, 8, Real, 16)       MATMUL_INSTANCE(Complex, 8, Complex, 16)
MATMUL_INSTANCE(Complex, 16, Integer, 1)    MATMUL_INSTANCE(Complex, 16, Integer, 2)
MATMUL_INSTANCE(Complex, 16, Integer, 4)    MATMUL_INSTANCE(Complex, 16, Integer, 8)
MATMUL_INSTANCE(Complex, 16, Real, 4)       MATMUL_INSTANCE(Complex, 16, Real, 8)
MATMUL_INSTANCE(Complex, 16, Real, 16)
MATMUL_INSTANCE(Complex, 16, Complex, 4)    MATMUL_INSTANCE(Complex, 16, Complex, 8)
MATMUL_INSTANCE(Complex, 16, Complex, 16)
MATMUL_DIRECT_INSTANCE(Integer, 1, Real, 16)       MATMUL_DIRECT_INSTANCE(Integer, 1, Complex, 16)
MATMUL_DIRECT_INSTANCE(Integer, 2, Real, 16)       MATMUL_DIRECT_INSTANCE(Integer, 2, Complex, 16)
MATMUL_DIRECT_INSTANCE(Integer, 4, Real, 16)       MATMUL_DIRECT_INSTANCE(Integer, 4, Complex, 16)
MATMUL_DIRECT_INSTANCE(Integer, 8, Real, 16)       MATMUL_DIRECT_INSTANCE(Integer, 8, Complex, 16)
MATMUL_DIRECT_INSTANCE(Real, 4, Real, 16)          MATMUL_DIRECT_INSTANCE(Real, 4, Complex, 16)
MATMUL_DIRECT_INSTANCE(Real, 8, Real, 16)          MATMUL_DIRECT_INSTANCE(Real, 8, Complex, 16)
MATMUL_DIRECT_INSTANCE(Real, 16, Integer, 1)       MATMUL_DIRECT_INSTANCE(Real, 16, Integer, 2)
MATMUL_DIRECT_INSTANCE(Real, 16, Integer, 4)       MATMUL_DIRECT_INSTANCE(Real, 16, Integer, 8)
MATMUL_DIRECT_INSTANCE(Real, 16, Real, 4)          MATMUL_DIRECT_INSTANCE(Real, 16, Real, 8)
MATMUL_DIRECT_INSTANCE(Real, 16, Real, 16)
MATMUL_DIRECT_INSTANCE(Real, 16, Complex, 4)       MATMUL_DIRECT_INSTANCE(Real, 16, Complex, 8)
MATMUL_DIRECT_INSTANCE(Real, 16, Complex, 16)
MATMUL_DIRECT_INSTANCE(Complex, 4, Real, 16)       MATMUL_DIRECT_INSTANCE(Complex, 4, Complex, 16)
MATMUL_DIRECT_INSTANCE(Complex, 8, Real, 16)       MATMUL_DIRECT_INSTANCE(Complex, 8, Complex, 16)
MATMUL_DIRECT_INSTANCE(Complex, 16, Integer, 1)    MATMUL_DIRECT_INSTANCE(Complex, 16, Integer, 2)
MATMUL_DIRECT_INSTANCE(Complex, 16, Integer, 4)    MATMUL_DIRECT_INSTANCE(Complex, 16, Integer, 8)
MATMUL_DIRECT_INSTANCE(Complex, 16, Real, 4)       MATMUL_DIRECT_INSTANCE(Complex, 16, Real, 8)
MATMUL_DIRECT_INSTANCE(Complex, 16, Real, 16)
MATMUL_DIRECT_INSTANCE(Complex, 16, Complex, 4)    MATMUL_DIRECT_INSTANCE(Complex, 16, Complex, 8)
MATMUL_DIRECT_INSTANCE(Complex, 16, Complex, 16)
#endif // HAS_LDBL128 || HAS_FLOAT128
    // clang-format on

    RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
