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
    return (char)OCKL_MANGLE_T(ctz,u8)((uchar)x);
}

UEXPATTR uchar
ctz(uchar x)
{
    return OCKL_MANGLE_T(ctz,u8)(x);
}

UEXPATTR short
ctz(short x)
{
    return (short)OCKL_MANGLE_T(ctz,u16)((ushort)x);
}

UEXPATTR ushort
ctz(ushort x)
{
    return OCKL_MANGLE_T(ctz,u16)(x);
}

UEXPATTR int
ctz(int x)
{
    return (uint)OCKL_MANGLE_U32(ctz)((uint)x);
}

UEXPATTR uint
ctz(uint x)
{
    return OCKL_MANGLE_U32(ctz)(x);
}

UEXPATTR long
ctz(long x)
{
    return (long)OCKL_MANGLE_U64(ctz)((ulong)x);
}

UEXPATTR ulong
ctz(ulong x)
{
    return OCKL_MANGLE_U64(ctz)(x);
}

