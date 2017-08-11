/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "int.h"

#define UEXPATTR __attribute__((always_inline, overloadable, const))
UEXP(char,clz)
UEXP(uchar,clz)
UEXP(short,clz)
UEXP(ushort,clz)
UEXP(int,clz)
UEXP(uint,clz)
UEXP(long,clz)
UEXP(ulong,clz)

UEXPATTR char
clz(char x)
{
    uint y = (uint)(uchar)x;
    uint z = __ockl_clz_u32(y);
    return (char)(z - 24u);
}

UEXPATTR uchar
clz(uchar x)
{
    uint y = (uint)x;
    uint z = __ockl_clz_u32(y);
    return (uchar)(z - 24u);
}

UEXPATTR short
clz(short x)
{
    uint y = (uint)(ushort)x;
    uint z = __ockl_clz_u32(y);
    return (short)(z - 16u);
}

UEXPATTR ushort
clz(ushort x)
{
    uint y = (uint)x;
    uint z = __ockl_clz_u32(y);
    return (ushort)(z - 16u);
}

UEXPATTR int
clz(int x)
{
    return (int)__ockl_clz_u32((uint)x);
}

UEXPATTR uint
clz(uint x)
{
    return __ockl_clz_u32(x);
}

UEXPATTR long
clz(long x)
{
    return (long)__ockl_clz_u64((ulong)x);
}

UEXPATTR ulong
clz(ulong x)
{
    return __ockl_clz_u64(x);
}

