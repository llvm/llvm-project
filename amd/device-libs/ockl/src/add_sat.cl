/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "ockl.h"

__attribute__((always_inline)) int
OCKL_MANGLE_I32(add_sat)(int x, int y)
{
    int s;
    bool c = __llvm_sadd_with_overflow_i32(x, y, &s);
    int lim = (x >> 31) ^ INT_MAX;
    return c ? lim : s;
}

__attribute__((always_inline)) uint
OCKL_MANGLE_U32(add_sat)(uint x, uint y)
{
    uint s;
    bool c = __llvm_uadd_with_overflow_i32(x, y, &s);
    return c ? UINT_MAX : s;
}

__attribute__((always_inline)) long
OCKL_MANGLE_I64(add_sat)(long x, long y)
{
    long s;
    bool c = __llvm_sadd_with_overflow_i64(x, y, &s);
    long lim = (x >> 63) ^ LONG_MAX;
    return c ? lim : s;
}

__attribute__((always_inline)) ulong
OCKL_MANGLE_U64(add_sat)(ulong x, ulong y)
{
    ulong s;
    bool c = __llvm_uadd_with_overflow_i64(x, y, &s);
    return c ? ULONG_MAX : s;
}

