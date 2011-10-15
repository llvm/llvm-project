//===------- bswapsi2 - Implement bswapsi2 --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern uint32_t __bswapsi2(uint32_t);
//
// Reverse all the bytes in a 32-bit integer.
//
.align 2
DEFINE_COMPILERRT_FUNCTION(__bswapsi2)
#if __ARM_ARCH_5TEJ__ || __ARM_ARCH_4T__
    // before armv6 does not have "rev" instruction
 	eor	r1, r0, r0, ror #16
 	bic	r1, r1, #0xff0000
 	mov	r1, r1, lsr #8
 	eor	r0, r1, r0, ror #8
#else
    rev r0, r0
#endif
    bx  lr
