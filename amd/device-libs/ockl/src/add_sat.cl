/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

int
OCKL_MANGLE_I32(add_sat)(int x, int y)
{
    int s;
    bool c = __builtin_sadd_overflow(x, y, &s);
    return c ? (x < 0 ? INT_MIN : INT_MAX) : s;
}

uint
OCKL_MANGLE_U32(add_sat)(uint x, uint y)
{
    uint s;
    bool c = __builtin_uadd_overflow(x, y, &s);
    return c ? UINT_MAX : s;
}

long
OCKL_MANGLE_I64(add_sat)(long x, long y)
{
    long s;
    bool c = __builtin_saddl_overflow(x, y, &s);
    return c ? (x < 0 ? LONG_MIN : LONG_MAX) : s;
}

ulong
OCKL_MANGLE_U64(add_sat)(ulong x, ulong y)
{
    ulong s;
    bool c = __builtin_uaddl_overflow(x, y, &s);
    return c ? ULONG_MAX : s;
}

