
#include "int.h"

#define UEXPATTR __attribute__((always_inline, overloadable, const))
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
    int y = (int)(uchar)x;
    return (char)min(__ockl_ctz_i32(y), 8);
}

UEXPATTR uchar
ctz(uchar x)
{
    int y = (int)x;
    return (uchar)min(__ockl_ctz_i32(y), 8);
}

UEXPATTR short
ctz(short x)
{
    int y = (int)(ushort)x;
    return (short)min(__ockl_ctz_i32(y), 16);
}

UEXPATTR ushort
ctz(ushort x)
{
    int y = (int)x;
    return (ushort)min(__ockl_ctz_i32(y), 16);
}

UEXPATTR int
ctz(int x)
{
    return __ockl_ctz_i32(x);
}

UEXPATTR uint
ctz(uint x)
{
    return __ockl_ctz_i32((int)x);
}

__attribute__((always_inline, const)) static ulong
ctz_u64(ulong x)
{
    int xlo = (int)x;
    int xhi = (int)(x >> 32);
    int zlo = __ockl_ctz_i32(xlo);
    int zhi = __ockl_ctz_i32(xhi) + 32;
    return (ulong)(xlo == 0 ? zhi : zlo);
}

extern __attribute__((overloadable, always_inline, const, alias("ctz_u64"))) ulong ctz(ulong);
extern __attribute__((overloadable, always_inline, const, alias("ctz_u64")))  long ctz(long);

