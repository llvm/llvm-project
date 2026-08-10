//===-- lib/runtime/matmul-transpose-extypes-real10.cpp ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Extended-type MATMUL-TRANSPOSE instances: Real10, Complex10.

#include "matmul-transpose.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

// clang-format off
#if HAS_FLOAT80
MATMUL_INSTANCE(Integer, 1, Real, 10)       MATMUL_INSTANCE(Integer, 1, Complex, 10)
MATMUL_INSTANCE(Integer, 2, Real, 10)       MATMUL_INSTANCE(Integer, 2, Complex, 10)
MATMUL_INSTANCE(Integer, 4, Real, 10)       MATMUL_INSTANCE(Integer, 4, Complex, 10)
MATMUL_INSTANCE(Integer, 8, Real, 10)       MATMUL_INSTANCE(Integer, 8, Complex, 10)
MATMUL_INSTANCE(Unsigned, 1, Real, 10)      MATMUL_INSTANCE(Unsigned, 1, Complex, 10)
MATMUL_INSTANCE(Unsigned, 2, Real, 10)      MATMUL_INSTANCE(Unsigned, 2, Complex, 10)
MATMUL_INSTANCE(Unsigned, 4, Real, 10)      MATMUL_INSTANCE(Unsigned, 4, Complex, 10)
MATMUL_INSTANCE(Unsigned, 8, Real, 10)      MATMUL_INSTANCE(Unsigned, 8, Complex, 10)
MATMUL_INSTANCE(Real, 4, Real, 10)          MATMUL_INSTANCE(Real, 4, Complex, 10)
MATMUL_INSTANCE(Real, 8, Real, 10)          MATMUL_INSTANCE(Real, 8, Complex, 10)
MATMUL_INSTANCE(Real, 10, Integer, 1)       MATMUL_INSTANCE(Real, 10, Integer, 2)
MATMUL_INSTANCE(Real, 10, Integer, 4)       MATMUL_INSTANCE(Real, 10, Integer, 8)
MATMUL_INSTANCE(Real, 10, Unsigned, 1)      MATMUL_INSTANCE(Real, 10, Unsigned, 2)
MATMUL_INSTANCE(Real, 10, Unsigned, 4)      MATMUL_INSTANCE(Real, 10, Unsigned, 8)
MATMUL_INSTANCE(Real, 10, Real, 4)          MATMUL_INSTANCE(Real, 10, Real, 8)
MATMUL_INSTANCE(Real, 10, Real, 10)
MATMUL_INSTANCE(Real, 10, Complex, 4)       MATMUL_INSTANCE(Real, 10, Complex, 8)
MATMUL_INSTANCE(Real, 10, Complex, 10)
MATMUL_INSTANCE(Complex, 4, Real, 10)       MATMUL_INSTANCE(Complex, 4, Complex, 10)
MATMUL_INSTANCE(Complex, 8, Real, 10)       MATMUL_INSTANCE(Complex, 8, Complex, 10)
MATMUL_INSTANCE(Complex, 10, Integer, 1)    MATMUL_INSTANCE(Complex, 10, Integer, 2)
MATMUL_INSTANCE(Complex, 10, Integer, 4)    MATMUL_INSTANCE(Complex, 10, Integer, 8)
MATMUL_INSTANCE(Complex, 10, Unsigned, 1)   MATMUL_INSTANCE(Complex, 10, Unsigned, 2)
MATMUL_INSTANCE(Complex, 10, Unsigned, 4)   MATMUL_INSTANCE(Complex, 10, Unsigned, 8)
MATMUL_INSTANCE(Complex, 10, Real, 4)       MATMUL_INSTANCE(Complex, 10, Real, 8)
MATMUL_INSTANCE(Complex, 10, Real, 10)
MATMUL_INSTANCE(Complex, 10, Complex, 4)    MATMUL_INSTANCE(Complex, 10, Complex, 8)
MATMUL_INSTANCE(Complex, 10, Complex, 10)
MATMUL_DIRECT_INSTANCE(Integer, 1, Real, 10)       MATMUL_DIRECT_INSTANCE(Integer, 1, Complex, 10)
MATMUL_DIRECT_INSTANCE(Integer, 2, Real, 10)       MATMUL_DIRECT_INSTANCE(Integer, 2, Complex, 10)
MATMUL_DIRECT_INSTANCE(Integer, 4, Real, 10)       MATMUL_DIRECT_INSTANCE(Integer, 4, Complex, 10)
MATMUL_DIRECT_INSTANCE(Integer, 8, Real, 10)       MATMUL_DIRECT_INSTANCE(Integer, 8, Complex, 10)
MATMUL_DIRECT_INSTANCE(Unsigned, 1, Real, 10)      MATMUL_DIRECT_INSTANCE(Unsigned, 1, Complex, 10)
MATMUL_DIRECT_INSTANCE(Unsigned, 2, Real, 10)      MATMUL_DIRECT_INSTANCE(Unsigned, 2, Complex, 10)
MATMUL_DIRECT_INSTANCE(Unsigned, 4, Real, 10)      MATMUL_DIRECT_INSTANCE(Unsigned, 4, Complex, 10)
MATMUL_DIRECT_INSTANCE(Unsigned, 8, Real, 10)      MATMUL_DIRECT_INSTANCE(Unsigned, 8, Complex, 10)
MATMUL_DIRECT_INSTANCE(Real, 4, Real, 10)          MATMUL_DIRECT_INSTANCE(Real, 4, Complex, 10)
MATMUL_DIRECT_INSTANCE(Real, 8, Real, 10)          MATMUL_DIRECT_INSTANCE(Real, 8, Complex, 10)
MATMUL_DIRECT_INSTANCE(Real, 10, Integer, 1)       MATMUL_DIRECT_INSTANCE(Real, 10, Integer, 2)
MATMUL_DIRECT_INSTANCE(Real, 10, Integer, 4)       MATMUL_DIRECT_INSTANCE(Real, 10, Integer, 8)
MATMUL_DIRECT_INSTANCE(Real, 10, Unsigned, 1)      MATMUL_DIRECT_INSTANCE(Real, 10, Unsigned, 2)
MATMUL_DIRECT_INSTANCE(Real, 10, Unsigned, 4)      MATMUL_DIRECT_INSTANCE(Real, 10, Unsigned, 8)
MATMUL_DIRECT_INSTANCE(Real, 10, Real, 4)          MATMUL_DIRECT_INSTANCE(Real, 10, Real, 8)
MATMUL_DIRECT_INSTANCE(Real, 10, Real, 10)
MATMUL_DIRECT_INSTANCE(Real, 10, Complex, 4)       MATMUL_DIRECT_INSTANCE(Real, 10, Complex, 8)
MATMUL_DIRECT_INSTANCE(Real, 10, Complex, 10)
MATMUL_DIRECT_INSTANCE(Complex, 4, Real, 10)       MATMUL_DIRECT_INSTANCE(Complex, 4, Complex, 10)
MATMUL_DIRECT_INSTANCE(Complex, 8, Real, 10)       MATMUL_DIRECT_INSTANCE(Complex, 8, Complex, 10)
MATMUL_DIRECT_INSTANCE(Complex, 10, Integer, 1)    MATMUL_DIRECT_INSTANCE(Complex, 10, Integer, 2)
MATMUL_DIRECT_INSTANCE(Complex, 10, Integer, 4)    MATMUL_DIRECT_INSTANCE(Complex, 10, Integer, 8)
MATMUL_DIRECT_INSTANCE(Complex, 10, Unsigned, 1)   MATMUL_DIRECT_INSTANCE(Complex, 10, Unsigned, 2)
MATMUL_DIRECT_INSTANCE(Complex, 10, Unsigned, 4)   MATMUL_DIRECT_INSTANCE(Complex, 10, Unsigned, 8)
MATMUL_DIRECT_INSTANCE(Complex, 10, Real, 4)       MATMUL_DIRECT_INSTANCE(Complex, 10, Real, 8)
MATMUL_DIRECT_INSTANCE(Complex, 10, Real, 10)
MATMUL_DIRECT_INSTANCE(Complex, 10, Complex, 4)    MATMUL_DIRECT_INSTANCE(Complex, 10, Complex, 8)
MATMUL_DIRECT_INSTANCE(Complex, 10, Complex, 10)

#if HAS_FLOAT128
MATMUL_INSTANCE(Real, 10, Real, 16)          MATMUL_INSTANCE(Real, 10, Complex, 16)
MATMUL_INSTANCE(Real, 16, Real, 10)          MATMUL_INSTANCE(Real, 16, Complex, 10)
MATMUL_INSTANCE(Complex, 10, Real, 16)       MATMUL_INSTANCE(Complex, 10, Complex, 16)
MATMUL_INSTANCE(Complex, 16, Real, 10)       MATMUL_INSTANCE(Complex, 16, Complex, 10)
MATMUL_DIRECT_INSTANCE(Real, 10, Real, 16)          MATMUL_DIRECT_INSTANCE(Real, 10, Complex, 16)
MATMUL_DIRECT_INSTANCE(Real, 16, Real, 10)          MATMUL_DIRECT_INSTANCE(Real, 16, Complex, 10)
MATMUL_DIRECT_INSTANCE(Complex, 10, Real, 16)       MATMUL_DIRECT_INSTANCE(Complex, 10, Complex, 16)
MATMUL_DIRECT_INSTANCE(Complex, 16, Real, 10)       MATMUL_DIRECT_INSTANCE(Complex, 16, Complex, 10)
#endif
#endif // HAS_FLOAT80
    // clang-format on

    RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
