
#include "int.h"

#define ATTR __attribute__((always_inline, overloadable, const))

#define char_shift 8
#define short_shift 16

#define GENN(N,T) \
ATTR T##N \
mul_hi(T##N x, T##N y) \
{ \
    return convert_##T##N(mul24(convert_int##N(x), convert_int##N(y)) >> T##_shift); \
} \
 \
ATTR u##T##N \
mul_hi(u##T##N x, u##T##N y) \
{ \
    return convert_u##T##N(mul24(convert_uint##N(x), convert_uint##N(y)) >> T##_shift); \
}

#define GEN(T) \
    GENN(16,T) \
    GENN(8,T) \
    GENN(4,T) \
    GENN(3,T) \
    GENN(2,T) \
    GENN(,T)

GEN(char)
GEN(short)

#define BEXPATTR ATTR
BEXP(int,mul_hi)
BEXP(uint,mul_hi)
BEXP(long,mul_hi)
BEXP(ulong,mul_hi)

BEXPATTR int
mul_hi(int x, int y)
{
    return (int)(((long)x * (long)y) >> 32);
}

BEXPATTR uint
mul_hi(uint x, uint y)
{
    return (uint)(((ulong)x * (ulong)y) >> 32);
}

BEXPATTR long
mul_hi(long x, long y)
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

BEXPATTR ulong
mul_hi(ulong x, ulong y)
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

