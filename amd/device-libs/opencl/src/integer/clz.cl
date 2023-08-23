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
    return (char)OCKL_MANGLE_T(clz,u8)((uchar)x);
}

UEXPATTR uchar
clz(uchar x)
{
    return OCKL_MANGLE_T(clz,u8)(x);
}

UEXPATTR short
clz(short x)
{
    return (short)OCKL_MANGLE_T(clz,u16)((ushort)x);
}

UEXPATTR ushort
clz(ushort x)
{
    return OCKL_MANGLE_T(clz,u16)(x);
}

UEXPATTR int
clz(int x)
{
    return (int)OCKL_MANGLE_U32(clz)((uint)x);
}

UEXPATTR uint
clz(uint x)
{
    return OCKL_MANGLE_U32(clz)(x);
}

UEXPATTR long
clz(long x)
{
    return (long)OCKL_MANGLE_U64(clz)((ulong)x);
}

UEXPATTR ulong
clz(ulong x)
{
    return OCKL_MANGLE_U64(clz)(x);
}

