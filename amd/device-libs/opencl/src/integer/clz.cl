/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "int.h"

#define UEXPATTR __attribute__((overloadable, const))
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
    return (char)__ockl_clz_u8((uchar)x);
}

UEXPATTR uchar
clz(uchar x)
{
    return __ockl_clz_u8(x);
}

UEXPATTR short
clz(short x)
{
    return (short)__ockl_clz_u16((ushort)x);
}

UEXPATTR ushort
clz(ushort x)
{
    return __ockl_clz_u16(x);
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

