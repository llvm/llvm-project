//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_VECSIZE_OR_1 == 1
#define __CLC_RETTYPE __CLC_INTN
#else
#define __CLC_RETTYPE __CLC_S_GENTYPE
#endif

#if __CLC_FPSIZE == 32
#define __CLC_SIGNBIT_MASK SIGNBIT_SP32
#elif __CLC_FPSIZE == 64
#define __CLC_SIGNBIT_MASK SIGNBIT_DP64
#elif __CLC_FPSIZE == 16
#define __CLC_SIGNBIT_MASK SIGNBIT_FP16
#else
#error "Invalid FP size"
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_RETTYPE __clc_signbit(__CLC_GENTYPE x) {
#if __CLC_VECSIZE_OR_1 == 1
  return (__CLC_INTN)((__CLC_AS_S_GENTYPE(x) &
                       (__CLC_S_GENTYPE)__CLC_SIGNBIT_MASK) != 0);
#else
  return (__CLC_AS_S_GENTYPE(x) & (__CLC_S_GENTYPE)__CLC_SIGNBIT_MASK) !=
                 (__CLC_S_GENTYPE)0
             ? (__CLC_S_GENTYPE)-1
             : (__CLC_S_GENTYPE)0;
#endif
}

#undef __CLC_RETTYPE
#undef __CLC_SIGNBIT_MASK
