//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 64
#define MIN_CONSTANT 0x1.fffffffffffffp-1
#elif __CLC_FPSIZE == 32
#define MIN_CONSTANT 0x1.fffffep-1f
#elif __CLC_FPSIZE == 16
#define MIN_CONSTANT 0x1.ffcp-1h
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_fract(__CLC_GENTYPE x,
                                                 private __CLC_GENTYPE *iptr) {
  *iptr = __clc_floor(x);
  __CLC_GENTYPE r = __clc_fmin(x - *iptr, MIN_CONSTANT);
  r = __clc_isinf(x) ? __CLC_FP_LIT(0.0) : r;
  r = __clc_isnan(x) ? x : r;
  return r;
}

#define __CLC_FRACT_DEF(addrspace)                                             \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_fract(                            \
      __CLC_GENTYPE x, addrspace __CLC_GENTYPE *iptr) {                        \
    __CLC_GENTYPE private_iptr;                                                \
    __CLC_GENTYPE ret = __clc_fract(x, &private_iptr);                         \
    *iptr = private_iptr;                                                      \
    return ret;                                                                \
  }

__CLC_FRACT_DEF(local);
__CLC_FRACT_DEF(global);
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_FRACT_DEF(generic);
#endif

#undef MIN_CONSTANT
