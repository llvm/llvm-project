//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE
__clc_tanpi_reduced_eval(__CLC_GENTYPE x, __CLC_INTN is_odd) {
  __CLC_GENTYPE s = x * x;

  __CLC_GENTYPE t = __clc_mad(s, __clc_mad(s, __clc_mad(s, __clc_mad(s,
            __clc_mad(s, __clc_mad(s,
                0x1.7d2bd4p+16f, 0x1.a4d306p+12f), 0x1.435004p+11f), 0x1.4b6926p+9f),
                0x1.451e22p+7f), 0x1.467a9cp+5f), 0x1.4abb6ap+3f);

  t = x * s * t;
  t = __clc_mad(x, 0x1.921fb6p+1f, t);

  __CLC_GENTYPE tr = __CLC_FP_LIT(-1.0) / t;

  return is_odd ? tr : t;
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN
__clc_tanpi_reduced_eval(__CLC_DOUBLEN x, __CLC_INTN is_odd) {
  __CLC_GENTYPE s = x * x;
  __CLC_GENTYPE t = __clc_mad(s, __clc_mad(s, __clc_mad(s, __clc_mad(s,
             __clc_mad(s, __clc_mad(s, __clc_mad(s, __clc_mad(s,
             __clc_mad(s, __clc_mad(s, __clc_mad(s, __clc_mad(s,
             __clc_mad(s,
                 0x1.3fad0a71ea6d1p+32, -0x1.11a76ac97377bp+30), 0x1.ba2bcaca6da1bp+27), -0x1.79e8e2d7aaf57p+22),
                 0x1.c1c1102e46eccp+21), 0x1.31291bbcb5588p+19), 0x1.486b2d6bb3db2p+17), 0x1.45be1b46ff156p+15),
                 0x1.45f61b419c746p+13), 0x1.45f311045a4ffp+11), 0x1.45f4739a998c7p+9), 0x1.45fff9b243050p+7),
                 0x1.466bc6775cf74p+5), 0x1.4abbce625be8bp+3);
  t = x * s * t;
  t = __clc_mad(x, 0x1.921fb54442d18p+1, t);

  __CLC_GENTYPE tr = __CLC_FP_LIT(-1.0) / t;

  return __CLC_CONVERT_LONGN(is_odd) ? tr : t;
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_HALFN
__clc_tanpi_reduced_eval(__CLC_HALFN x, __CLC_INTN is_odd) {
  __CLC_HALFN s = x * x;

  __CLC_HALFN t =
      __clc_mad(s, __clc_mad(s, 0x1.3d8p+8h, 0x1.fe4p+4h), 0x1.508p+3h);

  t = x * s * t;
  t = __clc_mad(x, 0x1.92p+1h, t);

  __CLC_HALFN tr = __CLC_FP_LIT(-1.0) / t;

  return __CLC_CONVERT_SHORTN(is_odd) ? tr : t;
}

#endif

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_tanpi(__CLC_GENTYPE x) {
  x = __clc_select(x, __CLC_GENTYPE_NAN,
                   __CLC_CONVERT_S_GENTYPE(__clc_isinf(x)));

  __CLC_GENTYPE absx = __clc_fabs(x);

  __CLC_GENTYPE reduced;
  __CLC_INTN n = __clc_piArgReductionS(&reduced, absx);

  __CLC_GENTYPE t = __clc_tanpi_reduced_eval(reduced, (n & 1) != 0);

  __CLC_S_GENTYPE n_1_or_2 = __CLC_CONVERT_S_GENTYPE((n == 1) || (n == 2));

  __CLC_S_GENTYPE flip = (n_1_or_2 && (reduced == __CLC_FP_LIT(0.0)))
                             ? __CLC_GENTYPE_S_SIGNBIT
                             : (__CLC_S_GENTYPE)0;

  __CLC_S_GENTYPE result = (__CLC_AS_S_GENTYPE(t) ^ flip) ^
                           (__CLC_AS_S_GENTYPE(x) & __CLC_GENTYPE_S_SIGNBIT);

  return __CLC_AS_GENTYPE(result);
}
