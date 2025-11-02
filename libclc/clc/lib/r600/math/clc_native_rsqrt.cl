//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>

_CLC_OVERLOAD _CLC_DEF float __clc_native_rsqrt(float x) {
  return __builtin_r600_recipsqrt_ieeef(x);
}

#define __CLC_FLOAT_ONLY
#define __CLC_FUNCTION __clc_native_rsqrt
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
