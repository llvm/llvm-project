//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// pi/180 = ~0.01745329251994329577 or 0x1.1df46a2529d39p-6 or 0x1.1df46ap-6F
#if __CLC_FPSIZE == 32
#define RADIAN_LITERAL 0x1.1df46ap-6F
#elif __CLC_FPSIZE == 64
#define RADIAN_LITERAL 0x1.1df46a2529d39p-6
#elif __CLC_FPSIZE == 16
#define RADIAN_LITERAL (half)0x1.1df46a2529d39p-6
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_radians(__CLC_GENTYPE degrees) {
  return RADIAN_LITERAL * degrees;
}

#undef RADIAN_LITERAL
