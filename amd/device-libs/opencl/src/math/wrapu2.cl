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

#define ATTR __attribute__((overloadable))

#define float_ssuf _f32
#define double_ssuf _f64
#define half_ssuf _f16
#define half_psuf _2f16

#define SNAME(F,T) C(__ocml_,C(F,T##_ssuf))
#define PNAME(F,T) C(__ocml_,C(F,T##_psuf))

#define SLST2(F,T) SNAME(F,T)(x.s0), SNAME(F,T)(x.s1)
#define SLST3(F,T) SNAME(F,T)(x.s0), SNAME(F,T)(x.s1), SNAME(F,T)(x.s2)
#define SLST4(F,T) SLST2(F,T), SNAME(F,T)(x.s2), SNAME(F,T)(x.s3)
#define SLST8(F,T) SLST4(F,T), SNAME(F,T)(x.s4), SNAME(F,T)(x.s5), SNAME(F,T)(x.s6), SNAME(F,T)(x.s7)
#define SLST16(F,T) SLST8(F,T), SNAME(F,T)(x.s8), SNAME(F,T)(x.s9), SNAME(F,T)(x.sa), SNAME(F,T)(x.sb), \
                                SNAME(F,T)(x.sc), SNAME(F,T)(x.sd), SNAME(F,T)(x.se), SNAME(F,T)(x.sf)

#define PLST3(F,T) PNAME(F,T)(x.s01), SNAME(F,T)(x.s2)
#define PLST4(F,T) PNAME(F,T)(x.s01), PNAME(F,T)(x.s23)
#define PLST8(F,T) PLST4(F,T), PNAME(F,T)(x.s45), PNAME(F,T)(x.s67)
#define PLST16(F,T) PLST8(F,T), PNAME(F,T)(x.s89), PNAME(F,T)(x.sab), PNAME(F,T)(x.scd), PNAME(F,T)(x.sef)

#define SWRAPN(N,F,OT,IT,ST) \
ATTR OT##N \
F(IT##N x) \
{ \
    return (OT##N) ( SLST##N(F,ST) ); \
}

#define PWRAPN(N,F,OT,IT,ST) \
ATTR OT##N \
F(IT##N x) \
{ \
    return (OT##N) ( PLST##N(F,ST) ); \
}

#define WRAP1(F,OT,IT,ST) \
ATTR OT \
F(IT x) \
{ \
    return SNAME(F,ST)(x); \
}

#define WRAP2(F,OT,IT,ST) \
ATTR OT##2 \
F(IT##2 x) \
{ \
    return PNAME(F,ST)(x); \
}

#define SWRAP(F,OT,IT,ST) \
    SWRAPN(16,F,OT,IT,ST) \
    SWRAPN(8,F,OT,IT,ST) \
    SWRAPN(4,F,OT,IT,ST) \
    SWRAPN(3,F,OT,IT,ST) \
    SWRAPN(2,F,OT,IT,ST) \
    WRAP1(F,OT,IT,ST)

#define PWRAP(F,OT,IT,ST) \
    PWRAPN(16,F,OT,IT,ST) \
    PWRAPN(8,F,OT,IT,ST) \
    PWRAPN(4,F,OT,IT,ST) \
    PWRAPN(3,F,OT,IT,ST) \
    WRAP2(F,OT,IT,ST) \
    WRAP1(F,OT,IT,ST)

SWRAP(ilogb,int,float,float)
SWRAP(ilogb,int,double,double)
PWRAP(ilogb,int,half,half)

SWRAP(nan,float,uint,float)
SWRAP(nan,double,ulong,double)
PWRAP(nan,half,ushort,half)

