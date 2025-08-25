//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <clc/internal/clc.h>

#define __CLC_BODY <clc_bitselect.inc>
#include <clc/integer/gentype.inc>

#define __CLC_FLOAT_BITSELECT(f_type, i_type, width)                           \
  _CLC_OVERLOAD _CLC_DEF f_type##width __clc_bitselect(                        \
      f_type##width x, f_type##width y, f_type##width z) {                     \
    return __clc_as_##f_type##width(__clc_bitselect(                           \
        __clc_as_##i_type##width(x), __clc_as_##i_type##width(y),              \
        __clc_as_##i_type##width(z)));                                         \
  }

__CLC_FLOAT_BITSELECT(float, uint, )
__CLC_FLOAT_BITSELECT(float, uint, 2)
__CLC_FLOAT_BITSELECT(float, uint, 3)
__CLC_FLOAT_BITSELECT(float, uint, 4)
__CLC_FLOAT_BITSELECT(float, uint, 8)
__CLC_FLOAT_BITSELECT(float, uint, 16)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__CLC_FLOAT_BITSELECT(double, ulong, )
__CLC_FLOAT_BITSELECT(double, ulong, 2)
__CLC_FLOAT_BITSELECT(double, ulong, 3)
__CLC_FLOAT_BITSELECT(double, ulong, 4)
__CLC_FLOAT_BITSELECT(double, ulong, 8)
__CLC_FLOAT_BITSELECT(double, ulong, 16)

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__CLC_FLOAT_BITSELECT(half, ushort, )
__CLC_FLOAT_BITSELECT(half, ushort, 2)
__CLC_FLOAT_BITSELECT(half, ushort, 3)
__CLC_FLOAT_BITSELECT(half, ushort, 4)
__CLC_FLOAT_BITSELECT(half, ushort, 8)
__CLC_FLOAT_BITSELECT(half, ushort, 16)

#endif
