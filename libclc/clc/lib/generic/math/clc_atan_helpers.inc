//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN
__clc_atan_reduced(__CLC_FLOATN v) {
  __CLC_FLOATN t = v * v;
  __CLC_FLOATN z = __clc_mad(t, __clc_mad(t, __clc_mad(t, __clc_mad(t,
            __clc_mad(t, __clc_mad(t, __clc_mad(t,
                0x1.5a54bp-9f, -0x1.f4b218p-7f), 0x1.53f67ep-5f), -0x1.2fa9aep-4f),
                0x1.b26364p-4f), -0x1.22c1ccp-3f), 0x1.99717ep-3f), -0x1.5554c4p-2f);

  z = __clc_mad(v, t * z, v);
  return z;
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN
__clc_atan_reduced(__CLC_DOUBLEN v) {
  __CLC_DOUBLEN t = v * v;
  __CLC_DOUBLEN z = __clc_mad(t, __clc_mad(t, __clc_mad(t, __clc_mad(t,
             __clc_mad(t, __clc_mad(t, __clc_mad(t, __clc_mad(t,
             __clc_mad(t, __clc_mad(t, __clc_mad(t, __clc_mad(t,
             __clc_mad(t, __clc_mad(t, __clc_mad(t, __clc_mad(t,
             __clc_mad(t, __clc_mad(t, __clc_mad(t,
                 0x1.ba404b5e68a13p-17, -0x1.3e260bd3237f4p-13), 0x1.b2bb069efb384p-11), -0x1.7952daf56de9bp-9),
                 0x1.d6d43a595c56fp-8), -0x1.c6ea4a57d9582p-7), 0x1.67e295f08b19fp-6), -0x1.e9ae6fc27006ap-6),
                 0x1.2c15b5711927ap-5), -0x1.59976e82d3ff0p-5), 0x1.82d5d6ef28734p-5), -0x1.ae5ce6a214619p-5),
                 0x1.e1bb48427b883p-5), -0x1.110e48b207f05p-4), 0x1.3b13657b87036p-4), -0x1.745d119378e4fp-4),
                 0x1.c71c717e1913cp-4), -0x1.2492492376b7dp-3), 0x1.99999999952ccp-3), -0x1.5555555555523p-2);
  z = __clc_mad(v, t*z, v);
  return z;
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_HALFN
__clc_atan_reduced(__CLC_HALFN v) {
  __CLC_HALFN t = v * v;
  __CLC_HALFN z = __clc_mad(
      t, __clc_mad(t, __clc_mad(t, 0x1.938p-6h, -0x1.7f4p-4h), 0x1.7dcp-3h),
      -0x1.54p-2h);
  return __clc_mad(t, v * z, v);
}

#endif
