/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "int.h"

#define UEXPATTR __attribute__((overloadable, const))
UEXP(char,ctz)
UEXP(uchar,ctz)
UEXP(short,ctz)
UEXP(ushort,ctz)
UEXP(int,ctz)
UEXP(uint,ctz)
UEXP(long,ctz)
UEXP(ulong,ctz)

UEXPATTR char
ctz(char x)
{
    return (char)__ockl_ctz_u8((uchar)x);
}

UEXPATTR uchar
ctz(uchar x)
{
    return __ockl_ctz_u8(x);
}

UEXPATTR short
ctz(short x)
{
    return (short)__ockl_ctz_u16((ushort)x);
}

UEXPATTR ushort
ctz(ushort x)
{
    return __ockl_ctz_u16(x);
}

UEXPATTR int
ctz(int x)
{
    return (int)__ockl_ctz_u32((uint)x);
}

UEXPATTR uint
ctz(uint x)
{
    return __ockl_ctz_u32(x);
}

UEXPATTR long
ctz(long x)
{
    return (long)__ockl_ctz_u64((ulong)x);
}

UEXPATTR ulong
ctz(ulong x)
{
    return __ockl_ctz_u64(x);
}

