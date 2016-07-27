/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "int.h"

#define TEXPATTR __attribute__((always_inline, overloadable, const))

TEXP(char,mad_sat)
TEXP(uchar,mad_sat)
TEXP(short,mad_sat)
TEXP(ushort,mad_sat)
TEXP(int,mad_sat)
TEXP(uint,mad_sat)
TEXP(long,mad_sat)
TEXP(ulong,mad_sat)

TEXPATTR char
mad_sat(char a, char b, char c)
{
    return (char)clamp(mad24((int)a, (int)b, (int)c), CHAR_MIN, CHAR_MAX);
}

TEXPATTR uchar
mad_sat(uchar a, uchar b, uchar c)
{
    return (uchar)min(mad24((uint)a, (uint)b, (uint)c), (uint)UCHAR_MAX);
}

TEXPATTR short
mad_sat(short a, short b, short c)
{
    return (short)clamp(mad24((int)a, (int)b, (int)c), SHRT_MIN, SHRT_MAX);
}

TEXPATTR ushort
mad_sat(ushort a, ushort b, ushort c)
{
    return (ushort)min(mad24((uint)a, (uint)b, (uint)c), (uint)USHRT_MAX);
}

TEXPATTR int
mad_sat(int a, int b, int c)
{
    long d = as_long((int2)(a * b, mul_hi(a, b))) + (long)c;
    return (int)clamp(d, (long)INT_MIN, (long)INT_MAX);
}

TEXPATTR uint
mad_sat(uint a, uint b, uint c)
{
    ulong d = as_ulong((uint2)(a * b, mul_hi(a, b))) + (ulong)c;
    return (uint)min(d, (ulong)UINT_MAX);
}

TEXPATTR long
mad_sat(long a, long b, long c)
{
    ulong a0 = (ulong)a & 0xffffffffUL;
    long a1 = a >> 32;
    ulong b0 = (ulong)b & 0xffffffffUL;
    long b1 = b >> 32;
    ulong s0 = a0*b0;
    long t = a1*b0 + (s0 >> 32);
    long s1 = t & 0xffffffffL;
    long s2 = t >> 32;
    s1 = a0*b1 + s1;
    long lo = (s1 << 32) | (s0 & 0xffffffffL);
    long hi = a1*b1 + s2 + (s1 >> 32);

    t = lo + c;
    hi += (c > 0L) & (0x7fffffffffffffffL - c < lo);
    hi -= (c < 1L) & ((long)0x8000000000000000L - c > lo);
    lo = t;

    lo = (hi < 0L) & ((hi != -1L) | (lo >= 0L)) ? 0x8000000000000000L : lo;
    lo = (hi >= 0L) & ((hi > 0L) | (lo < 0L)) ? 0x7fffffffffffffffL : lo;

    return lo;
}

TEXPATTR ulong
mad_sat(ulong a, ulong b, ulong c)
{
    ulong a0 = a & 0xffffffffUL;
    ulong a1 = a >> 32;
    ulong b0 = b & 0xffffffffUL;
    ulong b1 = b >> 32;
    ulong s0 = a0*b0;
    ulong t = a1*b0 + (s0 >> 32);
    ulong s1 = t & 0xffffffffUL;
    ulong s2 = t >> 32;
    s1 = a0*b1 + s1;
    ulong lo = (s1 << 32) | (s0 & 0xffffffffUL);
    ulong hi = a1*b1 + s2 + (s1 >> 32);

    t = lo + c;
    hi += 0xffffffffffffffffUL - c < lo;
    lo = t;

    return hi > 0UL ? 0xffffffffffffffffUL : lo;
}

