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

#define int_suf _f32
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

#define WRAPN(N,F,OT,IT) \
ATTR OT##N \
F(IT##N x) \
{ \
    return (OT##N) ( LIST##N(F,OT) ); \
}

#define WRAP1(F,OT,IT) \
ATTR OT \
F(IT x) \
{ \
    return ONAME(F,OT)(x); \
}

#define WRAP(F,OT,IT) \
    WRAPN(16,F,OT,IT) \
    WRAPN(8,F,OT,IT) \
    WRAPN(4,F,OT,IT) \
    WRAPN(3,F,OT,IT) \
    WRAPN(2,F,OT,IT) \
    WRAP1(F,OT,IT)

WRAP(ilogb,int,float)
WRAP(ilogb,int,double)
WRAP(ilogb,int,half)

WRAP(nan,float,uint)
WRAP(nan,double,ulong)
WRAP(nan,half,ushort)

