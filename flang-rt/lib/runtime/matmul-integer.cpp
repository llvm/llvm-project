//===-- lib/runtime/matmul-integer.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Integer*Integer and Logical*Logical MATMUL instances.

#include "matmul.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

// clang-format off
#define FOREACH_MATMUL_INTEGER_PAIR(macro) \
  macro(Integer, 1, Integer, 1)  macro(Integer, 1, Integer, 2) \
  macro(Integer, 1, Integer, 4)  macro(Integer, 1, Integer, 8) \
  macro(Integer, 2, Integer, 1)  macro(Integer, 2, Integer, 2) \
  macro(Integer, 2, Integer, 4)  macro(Integer, 2, Integer, 8) \
  macro(Integer, 4, Integer, 1)  macro(Integer, 4, Integer, 2) \
  macro(Integer, 4, Integer, 4)  macro(Integer, 4, Integer, 8) \
  macro(Integer, 8, Integer, 1)  macro(Integer, 8, Integer, 2) \
  macro(Integer, 8, Integer, 4)  macro(Integer, 8, Integer, 8)

FOREACH_MATMUL_INTEGER_PAIR(MATMUL_INSTANCE)
FOREACH_MATMUL_INTEGER_PAIR(MATMUL_DIRECT_INSTANCE)

#define FOREACH_MATMUL_LOGICAL_PAIR(macro) \
  macro(Logical, 1, Logical, 1)  macro(Logical, 1, Logical, 2) \
  macro(Logical, 1, Logical, 4)  macro(Logical, 1, Logical, 8) \
  macro(Logical, 2, Logical, 1)  macro(Logical, 2, Logical, 2) \
  macro(Logical, 2, Logical, 4)  macro(Logical, 2, Logical, 8) \
  macro(Logical, 4, Logical, 1)  macro(Logical, 4, Logical, 2) \
  macro(Logical, 4, Logical, 4)  macro(Logical, 4, Logical, 8) \
  macro(Logical, 8, Logical, 1)  macro(Logical, 8, Logical, 2) \
  macro(Logical, 8, Logical, 4)  macro(Logical, 8, Logical, 8)

FOREACH_MATMUL_LOGICAL_PAIR(MATMUL_INSTANCE)
FOREACH_MATMUL_LOGICAL_PAIR(MATMUL_DIRECT_INSTANCE)
// clang-format on

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
