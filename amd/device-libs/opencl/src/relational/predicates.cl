/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ocml.h"

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ATTR __attribute__((always_inline, overloadable, const))

#define _C(A,B) A##B
#define C(A,B) _C(A,B)

#define float_suff _f32
#define double_suff _f64
#define half_suff _f16

#define float_rtype int
#define double_rtype long
#define half_rtype short

#define ULIST2(F) -F(x.s0), -F(x.s1)
#define ULIST3(F) -F(x.s0), -F(x.s1), -F(x.s2)
#define ULIST4(F) ULIST2(F), -F(x.s2), -F(x.s3)
#define ULIST8(F) ULIST4(F), -F(x.s4), -F(x.s5), -F(x.s6), -F(x.s7)
#define ULIST16(F) ULIST8(F), -F(x.s8), -F(x.s9), -F(x.sa), -F(x.sb), -F(x.sc), -F(x.sd), -F(x.se), -F(x.sf)

#define UGENTN(N,F,T) \
ATTR C(T##_rtype,N) \
F(T##N x) \
{ \
    return (C(T##_rtype,N)) ( ULIST##N(F) ); \
}

#define UGENTS(F,T) \
ATTR int \
F(T x) \
{ \
    return C(__ocml_,C(F,T##_suff))(x); \
}

#define UGENT(F,T) \
    UGENTN(16,F,T) \
    UGENTN(8,F,T) \
    UGENTN(4,F,T) \
    UGENTN(3,F,T) \
    UGENTN(2,F,T) \
    UGENTS(F,T)

#define UGEN(F) \
    UGENT(F,float) \
    UGENT(F,double) \
    UGENT(F,half)

UGEN(isfinite)
UGEN(isinf)
UGEN(isnan)
UGEN(isnormal)
UGEN(signbit)

#define BGENTN(N,F,T,E) \
ATTR C(T##_rtype,N) \
F(T##N x, T##N y) \
{ \
    return E; \
}

#define BGENTS(F,T,E) \
ATTR int \
F(T x, T y) \
{ \
    return E; \
}

#define BGENT(F,T,E) \
    BGENTN(16,F,T,E) \
    BGENTN(8,F,T,E) \
    BGENTN(4,F,T,E) \
    BGENTN(3,F,T,E) \
    BGENTN(2,F,T,E) \
    BGENTS(F,T,E)

#define BGEN(F,E) \
    BGENT(F,float,E) \
    BGENT(F,double,E) \
    BGENT(F,half,E)

BGEN(isequal,x==y)
BGEN(isnotequal,x!=y)
BGEN(isgreater,x>y)
BGEN(isgreaterequal,x>=y)
BGEN(isless,x<y)
BGEN(islessequal,x<=y)

BGEN(isordered,!isunordered(x,y))
BGEN(isunordered,isnan(x)|isnan(y))
BGEN(islessgreater,(x<y)|(y<x))

