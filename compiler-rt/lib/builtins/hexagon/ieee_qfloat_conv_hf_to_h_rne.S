//===-- ieee_qfloat_conv_hf_to_h_rne.S - HVX hf-to-int16 conversion ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Half-float to signed int16 conversion with IEEE round-to-nearest-even.
// Uses QFloat (qf16) as an intermediate format on Hexagon V79.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

#if __HEXAGON_ARCH__ == 79
.p2align 5
DEFINE_COMPILERRT_FUNCTION(__qf_convert_hf_to_h_rne)
{
	R1=#0x3800 //+.5
	R2=#0xB800 //-.5
	V7.h=V0.hf //Convert and truncate half precision
}
{
	V6=#0
	R3=#0x00010001
	R4=#0x37FF
}
{
	V10=vsplat(R3)
	V12.h=vsplat(R4)
	V1.hf=V7.h //Convert truncated back to hf
}
{
	V3.h=vsplat(R1) //V3 = +.5
	V4.h=vsplat(R2) //V4 = -.5
	V5.qf16=vsub(V0.hf,V1.hf) //Sub truncated value from original value
	R0=#0x7FFF //Mask off sign bit of IEEE-754
}
{
	Q0=vcmp.gt(V0.hf,V6.hf) // hf value is pos (hf > 0)
	V5.hf=V5.qf16 //V5 is the fractional of original value
	V2.h=vsplat(R0) //Splat R0 to Mask off sign bit
}
{
	V1=vmux(Q0,V3,V4) //If input is pos value, use +.5. If neg value, use -.5 (Q in prev packet)
	V2=vand(V2,V5) //V2=abs(frac)
	V13=vand(V0,V2) //V13 = abs(input)
}
{
	Q1=vcmp.eq(V2.h,V3.h) // frac == .5 (0x3800)
	V1=vsplat(R3) //V1=1
	V4.qf16=vadd(V0.hf,V1.hf) // Add +.5 or -.5 to V0
	Q3=vcmp.eq(V13.h,V12.h) //Q3 = if(input == abs(.4998))
}
{
	V11=vand(V7,V10) //Mask off all the lower bits
	V3.h = vadd(V1.h,V1.h) //V3 is 2
}
{
	Q2=vcmp.eq(V11.h,V10.h) //Is input value odd if halfword equals 0x1
	V4.hf=V4.qf16 //hf value after add/sub .5
}
{
	Q1=and(Q1,!Q2) //Is frac .5 and input is even
	if (Q0) V1.h-=V3.h //If input is pos, sub 2 so V1 is -1
	V4=vmux(Q3,V6,V4)
}
{
	V4.h=V4.hf //Truncate, V4 has converted value
	V1=vmux(Q1,V1,V6) //If frac == .5 and input is even, use +1/-1. Else 0
}
{
	V0.h = vadd(V4.h,V1.h) //Add +1/-1 or 0. V4 has the result
	jumpr LR
}
END_COMPILERRT_FUNCTION(__qf_convert_hf_to_h_rne)
#endif
