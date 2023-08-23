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
    sub_sat(T x, T y)                               \
    {                                               \
        T s;                                        \
        bool c = __builtin_sub_overflow(x, y, &s);  \
        return c ? (x < 0 ? T##_min : T##_max) : s; \
    }                                               \
                                                    \
    ATTR u##T                                       \
    sub_sat(u##T x, u##T y)                         \
    {                                               \
        u##T s;                                     \
        bool c = __builtin_sub_overflow(x, y, &s);  \
        return c ? 0 : s;                           \
    }

GENN(char)
GENN(short)

#define BEXPATTR __attribute__((overloadable))
BEXP(char,sub_sat)
BEXP(uchar,sub_sat)
BEXP(short,sub_sat)
BEXP(ushort,sub_sat)
BEXP(int,sub_sat)
BEXP(uint,sub_sat)
BEXP(long,sub_sat)
BEXP(ulong,sub_sat)

BEXPATTR int
sub_sat(int x, int y)
{
    return __ockl_sub_sat_i32(x, y);
}

BEXPATTR uint
sub_sat(uint x, uint y)
{
    return __ockl_sub_sat_u32(x, y);
}

BEXPATTR long
sub_sat(long x, long y)
{
    return __ockl_sub_sat_i64(x, y);
}

BEXPATTR ulong
sub_sat(ulong x, ulong y)
{
    return __ockl_sub_sat_u64(x, y);
}

