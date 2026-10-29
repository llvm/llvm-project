//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/math/clc_get_twobypi_bits.h"

_CLC_OVERLOAD _CLC_DEF double __clc_get_twobypi_bits(double x, int y) {
  return __builtin_amdgcn_trig_preop(x, y);
}

#define __CLC_DOUBLE_ONLY
#define __CLC_ARG2_SCALAR_TYPE int
#define __CLC_FUNCTION __clc_get_twobypi_bits
#define __CLC_IMPL_FUNCTION(x, y) __builtin_amdgcn_trig_preop(x, y)
#define __CLC_BODY "clc/shared/binary_def_scalarize_loop.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION
