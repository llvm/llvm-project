//===-- ltdf2vfp.S - Implement ltdf2vfp -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern double __ltdf2vfp(double a, double b);
//
// Returns one iff a < b and neither is NaN.
// Uses Darwin calling convention where double precision arguments are passsed 
// like in GPR pairs.
//
	.syntax unified
	.align 2
DEFINE_COMPILERRT_FUNCTION(__ltdf2vfp)
	vmov 	d6, r0, r1	// load r0/r1 pair in double register
	vmov 	d7, r2, r3	// load r2/r3 pair in double register
	vcmp.f64 d6, d7
	vmrs	apsr_nzcv, fpscr
	movmi	r0, #1		// set result register to 1 if equal
	movpl	r0, #0
	bx	lr
