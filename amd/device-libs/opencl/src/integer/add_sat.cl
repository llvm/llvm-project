/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "int.h"

#define ATTR __attribute__((overloadable, const))

#define char_min CHAR_MIN
#define char_max CHAR_MAX
#define short_min SHRT_MIN
#define short_max SHRT_MAX

#define uchar_max UCHAR_MAX
#define ushort_max USHRT_MAX

#define GENN(T)                                     \
    ATTR T                                          \
    add_sat(T x, T y)                               \
    {                                               \
        T s;                                        \
        bool c = __builtin_add_overflow(x, y, &s);  \
        return c ? (x < 0 ? T##_min : T##_max) : s; \
    }                                               \
                                                    \
    ATTR u##T                                       \
    add_sat(u##T x, u##T y)                         \
    {                                               \
        u##T s;                                     \
        bool c = __builtin_add_overflow(x, y, &s);  \
        return c ? u##T##_max : s;                  \
    }

GENN(char)
GENN(short)

#define BEXPATTR __attribute__((overloadable))
BEXP(char,add_sat)
BEXP(uchar,add_sat)
BEXP(short,add_sat)
BEXP(ushort,add_sat)
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

