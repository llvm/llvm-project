//===-- floatunssisfvfp.S - Implement floatunssisfvfp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern float __floatunssisfvfp(unsigned int a);
//
// Converts single precision float to a 32-bit int rounding towards zero.
// Uses Darwin calling convention where a single precision result is 
// return in a GPR..
//
	.syntax unified
	.align 2
DEFINE_COMPILERRT_FUNCTION(__floatunssisfvfp)
	vmov	s15, r0	       // move int to float register s15
	vcvt.f32.u32 s15, s15  // convert 32-bit int in s15 to float in s15
	vmov	r0, s15        // move s15 to result register
	bx	lr
