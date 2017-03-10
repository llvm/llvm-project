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

#define float_ssuf _f32
#define double_ssuf _f64
#define half_ssuf _f16
#define half_psuf _2f16

#define SNAME(F,T) C(__ocml_,C(F,T##_ssuf))
#define PNAME(F,T) C(__ocml_,C(F,T##_psuf))

#define SLST2(F,T) SNAME(F,T)(x.s0), SNAME(F,T)(x.s1)
#define SLST3(F,T) SLST2(F,T), SNAME(F,T)(x.s2)
#define SLST4(F,T) SLST2(F,T), SNAME(F,T)(x.s2), SNAME(F,T)(x.s3)
#define SLST8(F,T) SLST4(F,T), SNAME(F,T)(x.s4), SNAME(F,T)(x.s5), SNAME(F,T)(x.s6), SNAME(F,T)(x.s7)
#define SLST16(F,T) SLST8(F,T), SNAME(F,T)(x.s8), SNAME(F,T)(x.s9), SNAME(F,T)(x.sa), SNAME(F,T)(x.sb), \
                                SNAME(F,T)(x.sc), SNAME(F,T)(x.sd), SNAME(F,T)(x.se), SNAME(F,T)(x.sf)

#define PLST3(F,T) PNAME(F,T)(x.s01), SNAME(F,T)(x.s2)
#define PLST4(F,T) PNAME(F,T)(x.s01), PNAME(F,T)(x.s23)
#define PLST8(F,T) PLST4(F,T), PNAME(F,T)(x.s45), PNAME(F,T)(x.s67)
#define PLST16(F,T) PLST8(F,T), PNAME(F,T)(x.s89), PNAME(F,T)(x.sab), PNAME(F,T)(x.scd), PNAME(F,T)(x.sef)

#define SWRAPNT(N,F,T) \
ATTR T##N \
F(T##N x) \
{ \
    return (T##N) ( SLST##N(F,T) ); \
}

#define PWRAPNT(N,F,T) \
ATTR T##N \
F(T##N x) \
{ \
    return (T##N) ( PLST##N(F,T) ); \
}

#define WRAP1T(F,T) \
ATTR T \
F(T x) \
{ \
    return SNAME(F,T)(x); \
}

#define WRAP2T(F,T) \
ATTR T##2 \
F(T##2 x) \
{ \
    return PNAME(F,T)(x); \
}

#define SWRAPT(F,T) \
    SWRAPNT(16,F,T) \
    SWRAPNT(8,F,T) \
    SWRAPNT(4,F,T) \
    SWRAPNT(3,F,T) \
    SWRAPNT(2,F,T) \
    WRAP1T(F,T)

#define PWRAPT(F,T) \
    PWRAPNT(16,F,T) \
    PWRAPNT(8,F,T) \
    PWRAPNT(4,F,T) \
    PWRAPNT(3,F,T) \
    WRAP2T(F,T) \
    WRAP1T(F,T)

#if !defined USE_CLP
#define WRAP(F) \
    SWRAPT(F,float) \
    SWRAPT(F,double) \
    PWRAPT(F,half)
#else
#define WRAP(F) \
    WRAP1T(F,float) \
    WRAP1T(F,double) \
    WRAP1T(F,half) \
    WRAP2T(F,half)
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

