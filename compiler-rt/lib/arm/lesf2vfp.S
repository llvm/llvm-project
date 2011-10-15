//===-- lesf2vfp.S - Implement lesf2vfp -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern int __lesf2vfp(float a, float b);
//
// Returns one iff a <= b and neither is NaN.
// Uses Darwin calling convention where single precision arguments are passsed 
// like 32-bit ints
//
	.syntax unified
	.align 2
DEFINE_COMPILERRT_FUNCTION(__lesf2vfp)
	vmov	s14, r0     // move from GPR 0 to float register
	vmov	s15, r1     // move from GPR 1 to float register
	vcmp.f32 s14, s15
	vmrs	apsr_nzcv, fpscr
	movls	r0, #1      // set result register to 1 if equal
	movhi	r0, #0
	bx	lr

