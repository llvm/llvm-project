//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma OPENCL FP_CONTRACT OFF

_CLC_DEF _CLC_OVERLOAD __CLC_INTN __clc_argReductionS(private __CLC_HALFN *r_lo,
                                                      __CLC_HALFN x) {
  const __CLC_FLOATN twobypi = 0x1.45f306p-1f;
  const __CLC_FLOATN pb2_a = 0x1.92p+0f;
  const __CLC_FLOATN pb2_b = 0x1.fap-12f;
  const __CLC_FLOATN pb2_c = 0x1.54442ep-20f;

  __CLC_FLOATN x_float = __CLC_CONVERT_FLOATN(x);
  __CLC_FLOATN rint = __clc_rint(x_float * twobypi);

  __CLC_FLOATN result = __clc_mad(
      rint, -pb2_c, __clc_mad(rint, -pb2_b, __clc_mad(rint, -pb2_a, x_float)));

  *r_lo = __CLC_CONVERT_HALFN(result);
  return __CLC_CONVERT_INTN(rint) & 0x3;
}

_CLC_DEF _CLC_OVERLOAD __CLC_SINCOS_RET_GENTYPE
__clc_sincos_reduced_eval(__CLC_HALFN x) {
  __CLC_HALFN t = x * x;

  __CLC_SINCOS_RET_GENTYPE ret;
  ret.cos = __clc_mad(t, __clc_mad(t, 0x1.4b4p-5h, -0x1.ffcp-2h), 1.0h);
  ret.sin = __clc_mad(x, t * __clc_mad(t, 0x1.0bp-7h, -0x1.554p-3h), x);
  return ret;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_SINCOS_RET_GENTYPE
__clc_sincospi_reduced_eval(__CLC_HALFN x) {
  __CLC_HALFN t = x * x;
  __CLC_HALFN sx = __clc_mad(t, 0x1.b84p+0h, -0x1.46cp+2h);
  sx = x * t * sx;
  sx = __clc_mad(x, 0x1.92p+1h, sx);

  __CLC_HALFN cx = __clc_mad(t, 0x1.fbp+1h, -0x1.3bcp+2h);
  cx = __clc_mad(t, cx, 1.0h);

  __CLC_SINCOS_RET_GENTYPE ret;
  ret.cos = cx;
  ret.sin = sx;
  return ret;
}
