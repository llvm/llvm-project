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

#define SLST2(F,T) SNAME(F,T)(a.s0,b.s0,c.s0), SNAME(F,T)(a.s1,b.s1,c.s1)
#define SLST3(F,T) SNAME(F,T)(a.s0,b.s0,c.s0), SNAME(F,T)(a.s1,b.s1,c.s1), SNAME(F,T)(a.s2,b.s2,c.s2)
#define SLST4(F,T) SLST2(F,T), SNAME(F,T)(a.s2,b.s2,c.s2), SNAME(F,T)(a.s3,b.s3,c.s3)
#define SLST8(F,T) SLST4(F,T), SNAME(F,T)(a.s4,b.s4,c.s4), SNAME(F,T)(a.s5,b.s5,c.s5), \
                               SNAME(F,T)(a.s6,b.s6,c.s6), SNAME(F,T)(a.s7,b.s7,c.s7)
#define SLST16(F,T) SLST8(F,T), SNAME(F,T)(a.s8,b.s8,c.s8), SNAME(F,T)(a.s9,b.s9,c.s9), \
                                SNAME(F,T)(a.sa,b.sa,c.sa), SNAME(F,T)(a.sb,b.sb,c.sb), \
                                SNAME(F,T)(a.sc,b.sc,c.sc), SNAME(F,T)(a.sd,b.sd,c.sd), \
                                SNAME(F,T)(a.se,b.se,c.se), SNAME(F,T)(a.sf,b.sf,c.sf)

#define PLST3(F,T) PNAME(F,T)(a.s01,b.s01,c.s01), SNAME(F,T)(a.s2,b.s2,c.s2)
#define PLST4(F,T) PNAME(F,T)(a.s01,b.s01,c.s01), PNAME(F,T)(a.s23,b.s23,c.s23)
#define PLST8(F,T) PLST4(F,T), PNAME(F,T)(a.s45,b.s45,c.s45), PNAME(F,T)(a.s67,b.s67,c.s67)
#define PLST16(F,T) PLST8(F,T), PNAME(F,T)(a.s89,b.s89,c.s89), PNAME(F,T)(a.sab,b.sab,c.sab), \
                                PNAME(F,T)(a.scd,b.scd,c.scd), PNAME(F,T)(a.sef,b.sef,c.sef)

#define SWRAPNT(N,F,T) \
ATTR T##N \
F(T##N a, T##N b, T##N c) \
{ \
    return (T##N) ( SLST##N(F,T) ); \
}

#define PWRAPNT(N,F,T) \
ATTR T##N \
F(T##N a, T##N b, T##N c) \
{ \
    return (T##N) ( PLST##N(F,T) ); \
}

#define WRAP1T(F,T) \
ATTR T \
F(T a, T b, T c) \
{ \
    return SNAME(F,T)(a, b, c); \
}

#define WRAP2T(F,T) \
ATTR T##2 \
F(T##2 a, T##2 b, T##2 c) \
{ \
    return PNAME(F,T)(a, b, c); \
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

WRAP(fma)
WRAP(mad)

