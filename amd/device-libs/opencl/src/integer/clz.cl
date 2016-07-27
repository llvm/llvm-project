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

__attribute__((always_inline, const)) static ulong
clz_u64(ulong x)
{
    uint xlo = (uint)x;
    uint xhi = (uint)(x >> 32);
    uint zlo = __ockl_clz_u32(xlo) + 32u;
    uint zhi = __ockl_clz_u32(xhi);
    return (ulong)(xhi == 0 ? zlo : zhi);
}

extern __attribute__((overloadable, always_inline, const, alias("clz_u64"))) ulong clz(ulong);
extern __attribute__((overloadable, always_inline, const, alias("clz_u64")))  long clz(long);

