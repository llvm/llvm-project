/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "int.h"
#include "irif.h"

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
    return (char)BUILTIN_CTZ_U8((uchar)x);
}

UEXPATTR uchar
ctz(uchar x)
{
    return BUILTIN_CTZ_U8(x);
}

UEXPATTR short
ctz(short x)
{
    return (short)BUILTIN_CTZ_U16((ushort)x);
}

UEXPATTR ushort
ctz(ushort x)
{
    return BUILTIN_CTZ_U16(x);
}

UEXPATTR int
ctz(int x)
{
    return (int)BUILTIN_CTZ_U32((uint)x);
}

UEXPATTR uint
ctz(uint x)
{
    return BUILTIN_CTZ_U32(x);
}

UEXPATTR long
ctz(long x)
{
    return (long)BUILTIN_CTZ_U64((ulong)x);
}

UEXPATTR ulong
ctz(ulong x)
{
    return BUILTIN_CTZ_U64(x);
}

