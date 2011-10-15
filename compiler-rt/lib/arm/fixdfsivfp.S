//===-- fixdfsivfp.S - Implement fixdfsivfp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern int __fixdfsivfp(double a);
//
// Converts double precision float to a 32-bit int rounding towards zero.
// Uses Darwin calling convention where a double precision parameter is 
// passed in GPR register pair.
//
	.syntax unified
	.align 2
DEFINE_COMPILERRT_FUNCTION(__fixdfsivfp)
	vmov	d7, r0, r1    // load double register from R0/R1
	vcvt.s32.f64 s15, d7  // convert double to 32-bit int into s15
	vmov	r0, s15	      // move s15 to result register
	bx	lr
