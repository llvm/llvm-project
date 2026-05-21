//===-- lib/runtime/matmul-transpose-real.cpp -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Real-involving MATMUL-TRANSPOSE instances.

#include "matmul-transpose.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

// clang-format off
#define FOREACH_MATMUL_REAL_PAIR(macro) \
  macro(Integer, 1, Real, 4)    macro(Integer, 1, Real, 8) \
  macro(Integer, 2, Real, 4)    macro(Integer, 2, Real, 8) \
  macro(Integer, 4, Real, 4)    macro(Integer, 4, Real, 8) \
  macro(Integer, 8, Real, 4)    macro(Integer, 8, Real, 8) \
  macro(Unsigned, 1, Real, 4)   macro(Unsigned, 1, Real, 8) \
  macro(Unsigned, 2, Real, 4)   macro(Unsigned, 2, Real, 8) \
  macro(Unsigned, 4, Real, 4)   macro(Unsigned, 4, Real, 8) \
  macro(Unsigned, 8, Real, 4)   macro(Unsigned, 8, Real, 8) \
  macro(Real, 4, Integer, 1)    macro(Real, 4, Integer, 2) \
  macro(Real, 4, Integer, 4)    macro(Real, 4, Integer, 8) \
  macro(Real, 8, Integer, 1)    macro(Real, 8, Integer, 2) \
  macro(Real, 8, Integer, 4)    macro(Real, 8, Integer, 8) \
  macro(Real, 4, Unsigned, 1)   macro(Real, 4, Unsigned, 2) \
  macro(Real, 4, Unsigned, 4)   macro(Real, 4, Unsigned, 8) \
  macro(Real, 8, Unsigned, 1)   macro(Real, 8, Unsigned, 2) \
  macro(Real, 8, Unsigned, 4)   macro(Real, 8, Unsigned, 8) \
  macro(Real, 4, Real, 4)       macro(Real, 4, Real, 8) \
  macro(Real, 8, Real, 4)       macro(Real, 8, Real, 8)

FOREACH_MATMUL_REAL_PAIR(MATMUL_INSTANCE)
FOREACH_MATMUL_REAL_PAIR(MATMUL_DIRECT_INSTANCE)
// clang-format on

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
