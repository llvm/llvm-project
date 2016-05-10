
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

#define LIST2S(F,T) ONAME(F,T)(x.s0,y), ONAME(F,T)(x.s1,y)
#define LIST3S(F,T) ONAME(F,T)(x.s0,y), ONAME(F,T)(x.s1,y), ONAME(F,T)(x.s2,y)
#define LIST4S(F,T) LIST2S(F,T), ONAME(F,T)(x.s2,y), ONAME(F,T)(x.s3,y)
#define LIST8S(F,T) LIST4S(F,T), ONAME(F,T)(x.s4,y), ONAME(F,T)(x.s5,y), ONAME(F,T)(x.s6,y), ONAME(F,T)(x.s7,y)
#define LIST16S(F,T) LIST8S(F,T), ONAME(F,T)(x.s8,y), ONAME(F,T)(x.s9,y), ONAME(F,T)(x.sa,y), ONAME(F,T)(x.sb,y), \
                                ONAME(F,T)(x.sc,y), ONAME(F,T)(x.sd,y), ONAME(F,T)(x.se,y), ONAME(F,T)(x.sf,y)

#define WRAPTN(N,F,TX,TY) \
ATTR TX##N \
F(TX##N x, TY##N y) \
{ \
    return (TX##N) ( LIST##N(F,TX) ); \
}

#define WRAPSTN(N,F,TX,TY) \
ATTR TX##N \
F(TX##N x, TY y) \
{ \
    return (TX##N) ( LIST##N##S(F,TX) ); \
}

#define WRAPT1(F,TX,TY) \
ATTR TX \
F(TX x, TY y) \
{ \
    return ONAME(F,TX)(x, y); \
}

#define WRAPT(F,TX,TY) \
    WRAPTN(16,F,TX,TY) \
    WRAPTN(8,F,TX,TY) \
    WRAPTN(4,F,TX,TY) \
    WRAPTN(3,F,TX,TY) \
    WRAPTN(2,F,TX,TY) \
    WRAPT1(F,TX,TY)

#define WRAPST(F,TX,TY) \
    WRAPTN(16,F,TX,TY) \
    WRAPSTN(16,F,TX,TY) \
    WRAPTN(8,F,TX,TY) \
    WRAPSTN(8,F,TX,TY) \
    WRAPTN(4,F,TX,TY) \
    WRAPSTN(4,F,TX,TY) \
    WRAPTN(3,F,TX,TY) \
    WRAPSTN(3,F,TX,TY) \
    WRAPTN(2,F,TX,TY) \
    WRAPSTN(2,F,TX,TY) \
    WRAPT1(F,TX,TY)

WRAPST(fmax,float,float)
WRAPST(fmax,double,double)
WRAPST(fmax,half,half)

WRAPST(fmin,float,float)
WRAPST(fmin,double,double)
WRAPST(fmin,half,half)

WRAPST(ldexp,float,int)
WRAPST(ldexp,double,int)
WRAPST(ldexp,half,int)

WRAPT(pown,float,int)
WRAPT(pown,double,int)
WRAPT(pown,half,int)

WRAPT(rootn,float,int)
WRAPT(rootn,double,int)
WRAPT(rootn,half,int)
