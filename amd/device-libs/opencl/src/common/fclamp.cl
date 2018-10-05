/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ATTR __attribute__((overloadable, const))

#define VLIST2 clamp(x.s0, lo.s0, hi.s0), clamp(x.s1, lo.s1, hi.s1)
#define VLIST3 VLIST2, clamp(x.s2, lo.s2, hi.s2)
#define VLIST4 VLIST3, clamp(x.s3, lo.s3, hi.s3)
#define VLIST8 VLIST4, clamp(x.s4, lo.s4, hi.s4),  clamp(x.s5, lo.s5, hi.s5),  clamp(x.s6, lo.s6, hi.s6),  clamp(x.s7, lo.s7, hi.s7)
#define VLIST16 VLIST8, clamp(x.s8, lo.s8, hi.s8),  clamp(x.s9, lo.s9, hi.s9),  clamp(x.sa, lo.sa, hi.sa),  clamp(x.sb, lo.sb, hi.sb), clamp(x.sc, lo.sc, hi.sc), clamp(x.sd, lo.sd, hi.sd), clamp(x.se, lo.se, hi.se), clamp(x.sf, lo.sf, hi.sf)

#define LIST2 clamp(x.s0, lo, hi), clamp(x.s1, lo, hi)
#define LIST3 LIST2, clamp(x.s2, lo, hi)
#define LIST4 LIST3, clamp(x.s3, lo, hi)
#define LIST8 LIST4, clamp(x.s4, lo, hi),  clamp(x.s5, lo, hi),  clamp(x.s6, lo, hi),  clamp(x.s7, lo, hi)
#define LIST16 LIST8, clamp(x.s8, lo, hi),  clamp(x.s9, lo, hi),  clamp(x.sa, lo, hi),  clamp(x.sb, lo, hi), clamp(x.sc, lo, hi), clamp(x.sd, lo, hi), clamp(x.se, lo, hi), clamp(x.sf, lo, hi)

#define GENN(N,T) \
ATTR T##N \
clamp(T##N x, T lo, T hi) \
{ \
    return (T##N)( LIST##N ); \
} \
 \
ATTR T##N \
clamp(T##N x, T##N lo, T##N hi) \
{ \
    return (T##N) ( VLIST##N ); \
}

#define GEN1(T) \
ATTR T \
clamp(T x, T lo, T hi) \
{ \
    return fmin(fmax(x, lo), hi); \
}

#define GEN(T) \
    GENN(16,T) \
    GENN(8,T) \
    GENN(4,T) \
    GENN(3,T) \
    GENN(2,T)

GEN(float)
GEN(double)
GEN(half)

ATTR float
clamp(float x, float lo, float hi)
{
    return __ockl_median3_f32(x, lo, hi);
}

ATTR double
clamp(double x, double lo, double hi)
{
    return fmin(fmax(x, lo), hi);
}

ATTR half
clamp(half x, half lo, half hi)
{
    return __ockl_median3_f16(x, lo, hi);
}

