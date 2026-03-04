/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "int.h"

#define ATTR __attribute__((overloadable, const))

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
    return __ockl_mul_hi_i32(x, y);
}

BEXPATTR uint
mul_hi(uint x, uint y)
{
    return __ockl_mul_hi_u32(x, y);
}

BEXPATTR long
mul_hi(long x, long y)
{
    return __ockl_mul_hi_i64(x, y);
}

BEXPATTR ulong
mul_hi(ulong x, ulong y)
{
    return __ockl_mul_hi_u64(x, y);
}

