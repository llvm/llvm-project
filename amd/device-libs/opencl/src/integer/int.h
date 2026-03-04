/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

#define ULIST2(F) F(x.s0), F(x.s1)
#define ULIST3(F) F(x.s0), F(x.s1), F(x.s2)
#define ULIST4(F) ULIST2(F), F(x.s2), F(x.s3)
#define ULIST8(F) ULIST4(F), F(x.s4), F(x.s5), F(x.s6), F(x.s7)
#define ULIST16(F) ULIST8(F), F(x.s8), F(x.s9), F(x.sa), F(x.sb), F(x.sc), F(x.sd), F(x.se), F(x.sf)

#define UEXPN(N,T,F) \
UEXPATTR T##N \
F(T##N x) \
{ \
    return (T##N) ( ULIST##N(F) ); \
}

#define UEXP(T,F) \
    UEXPN(16,T,F) \
    UEXPN(8,T,F) \
    UEXPN(4,T,F) \
    UEXPN(3,T,F) \
    UEXPN(2,T,F)

#define BLIST2(F) F(x.s0, y.s0), F(x.s1, y.s1)
#define BLIST3(F) F(x.s0, y.s0), F(x.s1, y.s1), F(x.s2, y.s2)
#define BLIST4(F) BLIST2(F), F(x.s2, y.s2), F(x.s3, y.s3)
#define BLIST8(F) BLIST4(F), F(x.s4, y.s4), F(x.s5, y.s5), F(x.s6, y.s6), F(x.s7, y.s7)
#define BLIST16(F) BLIST8(F), F(x.s8, y.s8), F(x.s9, y.s9), F(x.sa, y.sa), F(x.sb, y.sb), F(x.sc, y.sc), F(x.sd, y.sd), F(x.se, y.se), F(x.sf, y.sf)

#define BEXPN(N,T,F) \
BEXPATTR T##N \
F(T##N x, T##N y) \
{ \
    return (T##N) ( BLIST##N(F) ); \
}

#define BEXP(T,F) \
    BEXPN(16,T,F) \
    BEXPN(8,T,F) \
    BEXPN(4,T,F) \
    BEXPN(3,T,F) \
    BEXPN(2,T,F)

#define TLIST2(F) F(a.s0, b.s0, c.s0), F(a.s1, b.s1, c.s1)
#define TLIST3(F) F(a.s0, b.s0, c.s0), F(a.s1, b.s1, c.s1), F(a.s2, b.s2, c.s2)
#define TLIST4(F) TLIST2(F), F(a.s2, b.s2, c.s2), F(a.s3, b.s3, c.s3)
#define TLIST8(F) TLIST4(F), F(a.s4, b.s4, c.s4), F(a.s5, b.s5, c.s5), F(a.s6, b.s6, c.s6), F(a.s7, b.s7, c.s7)
#define TLIST16(F) TLIST8(F), F(a.s8, b.s8, c.s8), F(a.s9, b.s9, c.s9), F(a.sa, b.sa, c.sa), F(a.sb, b.sb, c.sb), F(a.sc, b.sc, c.sc), F(a.sd, b.sd, c.sd), F(a.se, b.se, c.se), F(a.sf, b.sf, c.sf)

#define TEXPN(N,T,F) \
TEXPATTR T##N \
F(T##N a, T##N b, T##N c) \
{ \
    return (T##N) ( TLIST##N(F) ); \
}

#define TEXP(T,F) \
    TEXPN(16,T,F) \
    TEXPN(8,T,F) \
    TEXPN(4,T,F) \
    TEXPN(3,T,F) \
    TEXPN(2,T,F)

static inline long
_gpu_mul_hi_i64(long x, long y)
{
    ulong x0 = (ulong)x & 0xffffffffUL;
    long x1 = x >> 32;
    ulong y0 = (ulong)y & 0xffffffffUL;
    long y1 = y >> 32;
    ulong z0 = x0*y0;
    long t = x1*y0 + (z0 >> 32);
    long z1 = t & 0xffffffffL;
    long z2 = t >> 32;
    z1 = x0*y1 + z1;
    return x1*y1 + z2 + (z1 >> 32);
}

static inline ulong
_gpu_mul_hi_u64(ulong x, ulong y)
{
    ulong x0 = x & 0xffffffffUL;
    ulong x1 = x >> 32;
    ulong y0 = y & 0xffffffffUL;
    ulong y1 = y >> 32;
    ulong z0 = x0*y0;
    ulong t = x1*y0 + (z0 >> 32);
    ulong z1 = t & 0xffffffffUL;
    ulong z2 = t >> 32;
    z1 = x0*y1 + z1;
    return x1*y1 + z2 + (z1 >> 32);
}

