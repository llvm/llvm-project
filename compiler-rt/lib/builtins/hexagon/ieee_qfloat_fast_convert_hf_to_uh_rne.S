//===-- ieee_qfloat_fast_convert_hf_to_uh_rne.S - HVX hf-to-uint16 conversion ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Half-float to unsigned int16 conversion with IEEE round-to-nearest-even.
// Uses QFloat (qf16) as an intermediate format on Hexagon V79. This is
// the most complex variant because unsigned int16 range [0, 65535] exceeds
// half-float precision, requiring a scale-by-0.5 trick.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

#if __HEXAGON_ARCH__ == 79
.p2align 5
DEFINE_COMPILERRT_FUNCTION(__qf_fast_convert_hf_to_uh_rne)
//V0.hf has value to be converted
{
	R1=#0x3800 //+.5
	V3=#0
}
{
	Q0=vcmp.gt(V0.hf,V3.hf) // hf value is pos (hf > 0)
	V1.h=vsplat(R1) //V1 = +.5
}
{
	V4.qf16=vadd(V0.hf,V1.hf) // Add +.5
}
{
	V4.hf=V4.qf16
}
{
	V4.h=V4.hf //Truncate, V4 has converted value
}
{
	V0=vmux(Q0,V4,V3) //If original pos value, use converted. Else 0
	jumpr LR
}
END_COMPILERRT_FUNCTION(__qf_fast_convert_hf_to_uh_rne)
#endif
