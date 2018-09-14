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

#define SLST2(F,T) SNAME(F,T)(x.s0,y.s0), SNAME(F,T)(x.s1,y.s1)
#define SLST3(F,T) SNAME(F,T)(x.s0,y.s0), SNAME(F,T)(x.s1,y.s1), SNAME(F,T)(x.s2,y.s2)
#define SLST4(F,T) SLST2(F,T), SNAME(F,T)(x.s2,y.s2), SNAME(F,T)(x.s3,y.s3)
#define SLST8(F,T) SLST4(F,T), SNAME(F,T)(x.s4,y.s4), SNAME(F,T)(x.s5,y.s5), SNAME(F,T)(x.s6,y.s6), SNAME(F,T)(x.s7,y.s7)
#define SLST16(F,T) SLST8(F,T), SNAME(F,T)(x.s8,y.s8), SNAME(F,T)(x.s9,y.s9), SNAME(F,T)(x.sa,y.sa), SNAME(F,T)(x.sb,y.sb), \
                                SNAME(F,T)(x.sc,y.sc), SNAME(F,T)(x.sd,y.sd), SNAME(F,T)(x.se,y.se), SNAME(F,T)(x.sf,y.sf)

#define SLST2S(F,T) SNAME(F,T)(x.s0,y), SNAME(F,T)(x.s1,y)
#define SLST3S(F,T) SNAME(F,T)(x.s0,y), SNAME(F,T)(x.s1,y), SNAME(F,T)(x.s2,y)
#define SLST4S(F,T) SLST2S(F,T), SNAME(F,T)(x.s2,y), SNAME(F,T)(x.s3,y)
#define SLST8S(F,T) SLST4S(F,T), SNAME(F,T)(x.s4,y), SNAME(F,T)(x.s5,y), SNAME(F,T)(x.s6,y), SNAME(F,T)(x.s7,y)
#define SLST16S(F,T) SLST8S(F,T), SNAME(F,T)(x.s8,y), SNAME(F,T)(x.s9,y), SNAME(F,T)(x.sa,y), SNAME(F,T)(x.sb,y), \
                                SNAME(F,T)(x.sc,y), SNAME(F,T)(x.sd,y), SNAME(F,T)(x.se,y), SNAME(F,T)(x.sf,y)

#define PLST3(F,T) PNAME(F,T)(x.s01,y.s01), SNAME(F,T)(x.s2,y.s2)
#define PLST4(F,T) PNAME(F,T)(x.s01,y.s01), PNAME(F,T)(x.s23,y.s23)
#define PLST8(F,T) PLST4(F,T), PNAME(F,T)(x.s45,y.s45), PNAME(F,T)(x.s67,y.s67)
#define PLST16(F,T) PLST8(F,T), PNAME(F,T)(x.s89,y.s89), PNAME(F,T)(x.sab,y.sab), PNAME(F,T)(x.scd,y.scd), PNAME(F,T)(x.sef,y.sef)

#define PLST3S(F,T) PNAME(F,T)(x.s01,yy), SNAME(F,T)(x.s2,y)
#define PLST4S(F,T) PNAME(F,T)(x.s01,yy), PNAME(F,T)(x.s23,yy)
#define PLST8S(F,T) PLST4S(F,T), PNAME(F,T)(x.s45,yy), PNAME(F,T)(x.s67,yy)
#define PLST16S(F,T) PLST8S(F,T), PNAME(F,T)(x.s89,yy), PNAME(F,T)(x.sab,yy), PNAME(F,T)(x.scd,yy), PNAME(F,T)(x.sef,yy)

#define SWRAPTN(N,F,TX,TY) \
ATTR TX##N \
F(TX##N x, TY##N y) \
{ \
    return (TX##N) ( SLST##N(F,TX) ); \
}

#define SWRAPSTN(N,F,TX,TY) \
ATTR TX##N \
F(TX##N x, TY y) \
{ \
    return (TX##N) ( SLST##N##S(F,TX) ); \
}

#define PWRAPTN(N,F,TX,TY) \
ATTR TX##N \
F(TX##N x, TY##N y) \
{ \
    return (TX##N) ( PLST##N(F,TX) ); \
}

#define PWRAPSTN(N,F,TX,TY) \
ATTR TX##N \
F(TX##N x, TY y) \
{ \
    TY##2 yy = (TY##2)y; \
    return (TX##N) ( PLST##N##S(F,TX) ); \
}

#define WRAPT1(F,TX,TY) \
ATTR TX \
F(TX x, TY y) \
{ \
    return SNAME(F,TX)(x, y); \
}

#define WRAPT2(F,TX,TY) \
ATTR TX##2 \
F(TX##2 x, TY##2 y) \
{ \
    return PNAME(F,TX)(x, y); \
}

#define WRAPT2S(F,TX,TY) \
ATTR TX##2 \
F(TX##2 x, TY y) \
{ \
    return PNAME(F,TX)(x, (TY##2)y); \
}

#define SWRAPT(F,TX,TY) \
    SWRAPTN(16,F,TX,TY) \
    SWRAPTN(8,F,TX,TY) \
    SWRAPTN(4,F,TX,TY) \
    SWRAPTN(3,F,TX,TY) \
    SWRAPTN(2,F,TX,TY) \
    WRAPT1(F,TX,TY)

#define SWRAPST(F,TX,TY) \
    SWRAPTN(16,F,TX,TY) \
    SWRAPSTN(16,F,TX,TY) \
    SWRAPTN(8,F,TX,TY) \
    SWRAPSTN(8,F,TX,TY) \
    SWRAPTN(4,F,TX,TY) \
    SWRAPSTN(4,F,TX,TY) \
    SWRAPTN(3,F,TX,TY) \
    SWRAPSTN(3,F,TX,TY) \
    SWRAPTN(2,F,TX,TY) \
    SWRAPSTN(2,F,TX,TY) \
    WRAPT1(F,TX,TY)

#define PWRAPT(F,TX,TY) \
    PWRAPTN(16,F,TX,TY) \
    PWRAPTN(8,F,TX,TY) \
    PWRAPTN(4,F,TX,TY) \
    PWRAPTN(3,F,TX,TY) \
    WRAPT2(F,TX,TY) \
    WRAPT1(F,TX,TY)

#define PWRAPST(F,TX,TY) \
    PWRAPTN(16,F,TX,TY) \
    PWRAPSTN(16,F,TX,TY) \
    PWRAPTN(8,F,TX,TY) \
    PWRAPSTN(8,F,TX,TY) \
    PWRAPTN(4,F,TX,TY) \
    PWRAPSTN(4,F,TX,TY) \
    PWRAPTN(3,F,TX,TY) \
    PWRAPSTN(3,F,TX,TY) \
    WRAPT2(F,TX,TY) \
    WRAPT2S(F,TX,TY) \
    WRAPT1(F,TX,TY)

SWRAPST(fmax,float,float)
SWRAPST(fmax,double,double)
PWRAPST(fmax,half,half)

SWRAPST(fmin,float,float)
SWRAPST(fmin,double,double)
PWRAPST(fmin,half,half)

SWRAPST(ldexp,float,int)
SWRAPST(ldexp,double,int)
PWRAPST(ldexp,half,int)

SWRAPST(max,float,float)
SWRAPST(max,double,double)
PWRAPST(max,half,half)

SWRAPST(min,float,float)
SWRAPST(min,double,double)
PWRAPST(min,half,half)

SWRAPT(pown,float,int)
SWRAPT(pown,double,int)
PWRAPT(pown,half,int)

SWRAPT(rootn,float,int)
SWRAPT(rootn,double,int)
PWRAPT(rootn,half,int)

