//===-- ieee_qfloat_fast_convert_hf_to_ub_rne.S - HVX hf-to-uint8 conversion ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Half-float to unsigned int8 conversion with IEEE round-to-nearest-even.
// Takes two HVX vectors of half-floats (V0, V1) and produces one vector
// of unsigned bytes. Uses QFloat (qf16) as an intermediate format on
// Hexagon V79.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

#if __HEXAGON_ARCH__ == 79
.p2align 5
DEFINE_COMPILERRT_FUNCTION(__qf_fast_convert_hf_to_ub_rne)
//V0.hf and V1.hf have value to be converted
{
	R1=#0x3800 //+.5
	V3=#0
}
{
	Q0=vcmp.gt(V0.hf,V3.hf) // hf value is pos (hf > 0)
	Q1=vcmp.gt(V1.hf,V3.hf) // hf value is pos (hf > 0)
	V2.h=vsplat(R1) //V1 = +.5
}
{
	V4.qf16=vadd(V0.hf,V2.hf) // Add +.5
}
{
	V5.qf16=vadd(V1.hf,V2.hf) // Add +.5
}
{
	V4.hf=V4.qf16
}
{
	V5.hf=V5.qf16
}
{
	V4.h=V4.hf //Truncate, V4 has converted value
}
{
	V5.h=V5.hf //Truncate, V5 has converted value
}
{
	V4=vmux(Q0,V4,V3) //If original pos value, use converted. Else 0
	V5=vmux(Q0,V5,V3) //If original pos value, use converted. Else 0
}
{
	V0.ub = vpack(V4.h,V5.h):sat
}
{
        V0.h = vshuff(V0.h)
        jumpr LR
}
END_COMPILERRT_FUNCTION(__qf_fast_convert_hf_to_ub_rne)
#endif
