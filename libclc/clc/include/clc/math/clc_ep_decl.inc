//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

typedef struct __CLC_XCONCAT(__clc_ep_pair_, __CLC_GENTYPE) {
  __CLC_GENTYPE lo, hi;
} __CLC_XCONCAT(__clc_ep_pair_, __CLC_GENTYPE);

#define __CLC_EP_PAIR __CLC_XCONCAT(__clc_ep_pair_, __CLC_GENTYPE)

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_absv(__CLC_EP_PAIR a);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_neg(__CLC_EP_PAIR a);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_conditional_sign_match(__CLC_EP_PAIR a, __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_make_pair(__CLC_GENTYPE a, __CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_add(__CLC_GENTYPE a, __CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_normalize(__CLC_EP_PAIR a);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_normalize_overflow(__CLC_EP_PAIR a);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_sub(__CLC_GENTYPE a, __CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_add(__CLC_GENTYPE a,
                                                              __CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sub(__CLC_GENTYPE a,
                                                              __CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_mul(__CLC_GENTYPE a,
                                                              __CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sqr(__CLC_GENTYPE a);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_add(__CLC_EP_PAIR a,
                                                              __CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_add(__CLC_EP_PAIR a, __CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_add(__CLC_GENTYPE a,
                                                              __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_add(__CLC_GENTYPE a, __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_add(__CLC_EP_PAIR a,
                                                              __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_add(__CLC_EP_PAIR a, __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sub(__CLC_EP_PAIR a,
                                                              __CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_sub(__CLC_EP_PAIR a, __CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sub(__CLC_GENTYPE a,
                                                              __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_sub(__CLC_GENTYPE a, __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sub(__CLC_EP_PAIR a,
                                                              __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_sub(__CLC_EP_PAIR a, __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_ldexp(__CLC_EP_PAIR a,
                                                                int e);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_mul(__CLC_EP_PAIR a,
                                                              __CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_mul_overflow(__CLC_EP_PAIR a, __CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_mul(__CLC_GENTYPE a,
                                                              __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_mul_overflow(__CLC_GENTYPE a, __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_mul(__CLC_EP_PAIR a,
                                                              __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_mul_overflow(__CLC_EP_PAIR a, __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_div(__CLC_GENTYPE a,
                                                              __CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_div(__CLC_EP_PAIR a,
                                                              __CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_div(__CLC_GENTYPE a,
                                                              __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_div(__CLC_EP_PAIR a, __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_div(__CLC_EP_PAIR a,
                                                              __CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_recip(__CLC_GENTYPE b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_recip(__CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_recip(__CLC_EP_PAIR b);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sqr(__CLC_EP_PAIR a);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sqrt(__CLC_GENTYPE a);

_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sqrt(__CLC_EP_PAIR a);

#if __CLC_FPSIZE == 32 || __CLC_FPSIZE == 64
_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_ep_exp(__CLC_EP_PAIR a);
_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_ln(__CLC_GENTYPE a);
#endif
