//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 64
#define NAN_MASK 0x7ff0000000000000ul
#elif __CLC_FPSIZE == 32
#define NAN_MASK 0x7fc00000
#elif __CLC_FPSIZE == 16
#define NAN_MASK 0x7e00
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_nan(__CLC_U_GENTYPE code) {
  const __CLC_U_GENTYPE mask = NAN_MASK;
  const __CLC_U_GENTYPE res = code | mask;
  return __CLC_AS_GENTYPE(res);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_nan(__CLC_S_GENTYPE code) {
  return __clc_nan(__CLC_AS_U_GENTYPE(code));
}

#undef NAN_MASK
