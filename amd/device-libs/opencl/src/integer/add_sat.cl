/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "int.h"

#define ATTR __attribute__((always_inline, overloadable, const))

#define char_lb CHAR_MIN
#define char_ub CHAR_MAX
#define char_max UCHAR_MAX
#define short_lb SHRT_MIN
#define short_ub SHRT_MAX
#define short_max USHRT_MAX

#define GENN(N,T) \
ATTR T##N \
add_sat(T##N x, T##N y) \
{ \
    int##N s = convert_int##N(x) + convert_int##N(y); \
    return convert_##T##N(clamp(s, (int##N) T##_lb, (int##N) T##_ub)); \
} \
 \
ATTR u##T##N \
add_sat(u##T##N x, u##T##N y) \
{ \
    uint##N s = convert_uint##N(x) + convert_uint##N(y); \
    return convert_u##T##N(min(s, (uint##N) T##_max)); \
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

#define BEXPATTR __attribute__((always_inline, overloadable))
BEXP(int,add_sat)
BEXP(uint,add_sat)
BEXP(long,add_sat)
BEXP(ulong,add_sat)

BEXPATTR int
add_sat(int x, int y)
{
    return __ockl_add_sat_i32(x, y);
}

BEXPATTR uint
add_sat(uint x, uint y)
{
    return __ockl_add_sat_u32(x, y);
}

BEXPATTR long
add_sat(long x, long y)
{
    return __ockl_add_sat_i64(x, y);
}

BEXPATTR ulong
add_sat(ulong x, ulong y)
{
    return __ockl_add_sat_u64(x, y);
}

