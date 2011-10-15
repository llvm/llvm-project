//===-- fixunssfsivfp.S - Implement fixunssfsivfp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern unsigned int __fixunssfsivfp(float a);
//
// Converts single precision float to a 32-bit unsigned int rounding towards 
// zero. All negative values become zero.
// Uses Darwin calling convention where a single precision parameter is 
// passed in a GPR..
//
	.syntax unified
	.align 2
DEFINE_COMPILERRT_FUNCTION(__fixunssfsivfp)
	vmov	s15, r0        // load float register from R0
	vcvt.u32.f32 s15, s15  // convert single to 32-bit unsigned into s15
	vmov	r0, s15	       // move s15 to result register
	bx	lr
