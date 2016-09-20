/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ocml.h"

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define _C(X,Y) X##Y
#define C(X,Y) _C(X,Y)

#define ATTR __attribute__((always_inline, overloadable))

#define float_suf _f32
#define double_suf _f64
#define half_suf _f16

#define ONAME(F,T) C(__ocml_,C(F,T##_suf))

#define LIST2(F,T) ONAME(F,T)(x.s0), ONAME(F,T)(x.s1)
#define LIST3(F,T) ONAME(F,T)(x.s0), ONAME(F,T)(x.s1), ONAME(F,T)(x.s2)
#define LIST4(F,T) LIST2(F,T), ONAME(F,T)(x.s2), ONAME(F,T)(x.s3)
#define LIST8(F,T) LIST4(F,T), ONAME(F,T)(x.s4), ONAME(F,T)(x.s5), ONAME(F,T)(x.s6), ONAME(F,T)(x.s7)
#define LIST16(F,T) LIST8(F,T), ONAME(F,T)(x.s8), ONAME(F,T)(x.s9), ONAME(F,T)(x.sa), ONAME(F,T)(x.sb), \
                                ONAME(F,T)(x.sc), ONAME(F,T)(x.sd), ONAME(F,T)(x.se), ONAME(F,T)(x.sf)

#define WRAPNT(N,F,T) \
ATTR T##N \
F(T##N x) \
{ \
    return (T##N) ( LIST##N(F,T) ); \
}

#define WRAP1T(F,T) \
ATTR T \
F(T x) \
{ \
    return ONAME(F,T)(x); \
}

#define WRAPT(F,T) \
    WRAPNT(16,F,T) \
    WRAPNT(8,F,T) \
    WRAPNT(4,F,T) \
    WRAPNT(3,F,T) \
    WRAPNT(2,F,T) \
    WRAP1T(F,T)

#if !defined USE_CLP
#define WRAP(F) \
    WRAPT(F,float) \
    WRAPT(F,double) \
    WRAPT(F,half)
#else
#define WRAP(F) \
    WRAP1T(F,float) \
    WRAP1T(F,double) \
    WRAP1T(F,half)
#endif

WRAP(acos)
WRAP(acosh)
WRAP(acospi)
WRAP(asin)
WRAP(asinh)
WRAP(asinpi)
WRAP(atan)
WRAP(atanh)
WRAP(atanpi)
WRAP(cbrt)
WRAP(ceil)
WRAP(cos)
WRAP(cosh)
WRAP(cospi)
WRAP(erfc)
WRAP(erf)
WRAP(exp)
WRAP(exp2)
WRAP(exp10)
WRAP(expm1)
WRAP(fabs)
WRAP(floor)
WRAP(lgamma)
WRAP(log)
WRAP(log2)
WRAP(log10)
WRAP(log1p)
WRAP(logb)
WRAP(rint)
WRAP(round)
WRAP(rsqrt)
WRAP(sin)
WRAP(sinh)
WRAP(sinpi)
WRAP(sqrt)
WRAP(tan)
WRAP(tanh)
WRAP(tanpi)
WRAP(tgamma)
WRAP(trunc)

