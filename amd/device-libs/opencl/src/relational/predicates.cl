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

#define float_ssuf _f32
#define double_ssuf _f64
#define half_ssuf _f16
#define half_psuf _2f16

#define float_rtype int
#define double_rtype long
#define half_rtype short

#define SNAME(F,T) C(__ocml_,C(F,T##_ssuf))
#define PNAME(F,T) C(__ocml_,C(F,T##_psuf))

#define USLST2(F,T) -SNAME(F,T)(x.s0), -SNAME(F,T)(x.s1)
#define USLST3(F,T) USLST2(F,T), -SNAME(F,T)(x.s2)
#define USLST4(F,T) USLST2(F,T), -SNAME(F,T)(x.s2), -SNAME(F,T)(x.s3)
#define USLST8(F,T) USLST4(F,T), -SNAME(F,T)(x.s4), -SNAME(F,T)(x.s5), -SNAME(F,T)(x.s6), -SNAME(F,T)(x.s7)
#define USLST16(F,T) USLST8(F,T), -SNAME(F,T)(x.s8), -SNAME(F,T)(x.s9), -SNAME(F,T)(x.sa), -SNAME(F,T)(x.sb), -SNAME(F,T)(x.sc), -SNAME(F,T)(x.sd), -SNAME(F,T)(x.se), -SNAME(F,T)(x.sf)

#define UPLST3(F,T) PNAME(F,T)(x.s01), -SNAME(F,T)(x.s2)
#define UPLST4(F,T) PNAME(F,T)(x.s01),  PNAME(F,T)(x.s23)
#define UPLST8(F,T) UPLST4(F,T), PNAME(F,T)(x.s45),  PNAME(F,T)(x.s67)
#define UPLST16(F,T) UPLST8(F,T), PNAME(F,T)(x.s89),  PNAME(F,T)(x.sab), PNAME(F,T)(x.scd),  PNAME(F,T)(x.sef)

#define USGENTN(N,F,T) \
ATTR C(T##_rtype,N) \
F(T##N x) \
{ \
    return (C(T##_rtype,N)) ( USLST##N(F,T) ); \
}

#define UPGENTN(N,F,T) \
ATTR C(T##_rtype,N) \
F(T##N x) \
{ \
    return (C(T##_rtype,N)) ( UPLST##N(F,T) ); \
}

#define UGENT1(F,T) \
ATTR int \
F(T x) \
{ \
    return SNAME(F,T)(x); \
}

#define UGENT2(F,T) \
ATTR C(T##_rtype,2) \
F(T##2 x) \
{ \
    return PNAME(F,T)(x); \
}

#define USGENT(F,T) \
    USGENTN(16,F,T) \
    USGENTN(8,F,T) \
    USGENTN(4,F,T) \
    USGENTN(3,F,T) \
    USGENTN(2,F,T) \
    UGENT1(F,T)

#define UPGENT(F,T) \
    UPGENTN(16,F,T) \
    UPGENTN(8,F,T) \
    UPGENTN(4,F,T) \
    UPGENTN(3,F,T) \
    UGENT2(F,T) \
    UGENT1(F,T)

#define UGEN(F) \
    USGENT(F,float) \
    USGENT(F,double) \
    UPGENT(F,half)

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

#define BGENT1(F,T,E) \
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
    BGENT1(F,T,E)

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

