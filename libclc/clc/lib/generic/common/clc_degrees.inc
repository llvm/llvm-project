//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// 180/pi = ~57.29577951308232087685 or 0x1.ca5dc1a63c1f8p+5 or 0x1.ca5dc2p+5F
#if __CLC_FPSIZE == 32
#define DEGREE_LITERAL 0x1.ca5dc2p+5F
#elif __CLC_FPSIZE == 64
#define DEGREE_LITERAL 0x1.ca5dc1a63c1f8p+5
#elif __CLC_FPSIZE == 16
#define DEGREE_LITERAL (half)0x1.ca5dc1a63c1f8p+5
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_degrees(__CLC_GENTYPE radians) {
  return DEGREE_LITERAL * radians;
}

#undef DEGREE_LITERAL
