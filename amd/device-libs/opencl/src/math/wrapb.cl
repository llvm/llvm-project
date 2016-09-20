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

#define LIST2(F,T) ONAME(F,T)(x.s0,y.s0), ONAME(F,T)(x.s1,y.s1)
#define LIST3(F,T) ONAME(F,T)(x.s0,y.s0), ONAME(F,T)(x.s1,y.s1), ONAME(F,T)(x.s2,y.s2)
#define LIST4(F,T) LIST2(F,T), ONAME(F,T)(x.s2,y.s2), ONAME(F,T)(x.s3,y.s3)
#define LIST8(F,T) LIST4(F,T), ONAME(F,T)(x.s4,y.s4), ONAME(F,T)(x.s5,y.s5), ONAME(F,T)(x.s6,y.s6), ONAME(F,T)(x.s7,y.s7)
#define LIST16(F,T) LIST8(F,T), ONAME(F,T)(x.s8,y.s8), ONAME(F,T)(x.s9,y.s9), ONAME(F,T)(x.sa,y.sa), ONAME(F,T)(x.sb,y.sb), \
                                ONAME(F,T)(x.sc,y.sc), ONAME(F,T)(x.sd,y.sd), ONAME(F,T)(x.se,y.se), ONAME(F,T)(x.sf,y.sf)

#define WRAPNT(N,F,T) \
ATTR T##N \
F(T##N x, T##N y) \
{ \
    return (T##N) ( LIST##N(F,T) ); \
}

#define WRAP1T(F,T) \
ATTR T \
F(T x, T y) \
{ \
    return ONAME(F,T)(x, y); \
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

WRAP(atan2)
WRAP(atan2pi)
WRAP(copysign)
WRAP(fdim)
WRAP(fmod)
WRAP(hypot)
WRAP(maxmag)
WRAP(minmag)
WRAP(nextafter)
WRAP(pow)
WRAP(powr)
WRAP(remainder)

