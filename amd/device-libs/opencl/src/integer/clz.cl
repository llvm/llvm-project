/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "int.h"
#include "irif.h"

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
    return (char)BUILTIN_CLZ_U8((uchar)x);
}

UEXPATTR uchar
clz(uchar x)
{
    return BUILTIN_CLZ_U8(x);
}

UEXPATTR short
clz(short x)
{
    return (short)BUILTIN_CLZ_U16((ushort)x);
}

UEXPATTR ushort
clz(ushort x)
{
    return BUILTIN_CLZ_U16(x);
}

UEXPATTR int
clz(int x)
{
    return (int)BUILTIN_CLZ_U32((uint)x);
}

UEXPATTR uint
clz(uint x)
{
    return BUILTIN_CLZ_U32(x);
}

UEXPATTR long
clz(long x)
{
    return (long)BUILTIN_CLZ_U64((ulong)x);
}

UEXPATTR ulong
clz(ulong x)
{
    return BUILTIN_CLZ_U64(x);
}

