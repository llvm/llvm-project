//===-- ieee_qfloat_fast_convert_hf_to_h_rne.S - HVX hf-to-int16 conversion ===//
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
DEFINE_COMPILERRT_FUNCTION(__qf_fast_convert_hf_to_h_rne)
//V0.hf has value to be converted
{
	R1=#0x3800 //+.5
	R2=#0xB800 //-.5
	V1=#0
}
{
	Q0=vcmp.gt(V0.hf,V1.hf) // hf value is pos (hf > 0)
	V1.h=vsplat(R1) //V1 = +.5
	V2.h=vsplat(R2) //V2 = -.5
}
{
	V1=vmux(Q0,V1,V2) //If pos value, use +.5. If neg value, use -.5
}
{
	V4.qf16=vadd(V0.hf,V1.hf) // Add +.5 or -.5 to V0
}
{
	V4.hf=V4.qf16
}
{
	V0.h=V4.hf //Truncate, V4 has converted value
	jumpr LR
}
END_COMPILERRT_FUNCTION(__qf_fast_convert_hf_to_h_rne)
#endif
