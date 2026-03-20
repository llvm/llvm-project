//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 16
#define __CLC_GENTYPE_EXPBITS (__CLC_S_GENTYPE) EXPBITS_FP16
#define __CLC_GENTYPE_MANTBITS (__CLC_S_GENTYPE) MANTBITS_FP16
#define __CLC_GENTYPE_SIGNBIT (__CLC_S_GENTYPE) SIGNBIT_FP16
#elif __CLC_FPSIZE == 32
#define __CLC_GENTYPE_EXPBITS (__CLC_S_GENTYPE) EXPBITS_SP32
#define __CLC_GENTYPE_MANTBITS (__CLC_S_GENTYPE) MANTBITS_SP32
#define __CLC_GENTYPE_SIGNBIT (__CLC_S_GENTYPE) SIGNBIT_SP32
#elif __CLC_FPSIZE == 64
#define __CLC_GENTYPE_EXPBITS (__CLC_S_GENTYPE) EXPBITS_DP64
#define __CLC_GENTYPE_MANTBITS (__CLC_S_GENTYPE) MANTBITS_DP64
#define __CLC_GENTYPE_SIGNBIT (__CLC_S_GENTYPE) SIGNBIT_DP64
#endif

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_flush_if_daz(__CLC_GENTYPE x) {
  if (!__CLC_GENTYPE_DENORMS_ARE_ZERO)
    return x;

  // Hack around canonicalize not working on spirv.
#if defined(CLC_SPIRV) || defined(CLC_CLSPV)
  __CLC_S_GENTYPE is_denorm_or_zero = __clc_fabs(x) < __CLC_GENTYPE_MIN;
  return __clc_select(x, __clc_copysign(__CLC_FP_LIT(0.0), x),
                      is_denorm_or_zero);
#else
  return __clc_canonicalize(x);
#endif
}

#undef __CLC_GENTYPE_EXPBITS
#undef __CLC_GENTYPE_MANTBITS
#undef __CLC_GENTYPE_SIGNBIT
