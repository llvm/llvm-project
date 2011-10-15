//===-- negsf2vfp.S - Implement negsf2vfp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern float __negsf2vfp(float a);
//
// Returns the negation of a single precision floating point numbers using the 
// Darwin calling convention where single arguments are passsed like 32-bit ints
//
	.syntax unified
	.align 2
DEFINE_COMPILERRT_FUNCTION(__negsf2vfp)
	eor	r0, r0, #-2147483648	// flip sign bit on float in r0
	bx	lr
