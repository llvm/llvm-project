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

#define LIST2(F,T) ONAME(F,T)(a.s0,b.s0,c.s0), ONAME(F,T)(a.s1,b.s1,c.s1)
#define LIST3(F,T) ONAME(F,T)(a.s0,b.s0,c.s0), ONAME(F,T)(a.s1,b.s1,c.s1), ONAME(F,T)(a.s2,b.s2,c.s2)
#define LIST4(F,T) LIST2(F,T), ONAME(F,T)(a.s2,b.s2,c.s2), ONAME(F,T)(a.s3,b.s3,c.s3)
#define LIST8(F,T) LIST4(F,T), ONAME(F,T)(a.s4,b.s4,c.s4), ONAME(F,T)(a.s5,b.s5,c.s5), \
                               ONAME(F,T)(a.s6,b.s6,c.s6), ONAME(F,T)(a.s7,b.s7,c.s7)
#define LIST16(F,T) LIST8(F,T), ONAME(F,T)(a.s8,b.s8,c.s8), ONAME(F,T)(a.s9,b.s9,c.s9), \
                                ONAME(F,T)(a.sa,b.sa,c.sa), ONAME(F,T)(a.sb,b.sb,c.sb), \
                                ONAME(F,T)(a.sc,b.sc,c.sc), ONAME(F,T)(a.sd,b.sd,c.sd), \
                                ONAME(F,T)(a.se,b.se,c.se), ONAME(F,T)(a.sf,b.sf,c.sf)

#define WRAPNT(N,F,T) \
ATTR T##N \
F(T##N a, T##N b, T##N c) \
{ \
    return (T##N) ( LIST##N(F,T) ); \
}

#define WRAP1T(F,T) \
ATTR T \
F(T a, T b, T c) \
{ \
    return ONAME(F,T)(a, b, c); \
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

WRAP(fma)
WRAP(mad)

