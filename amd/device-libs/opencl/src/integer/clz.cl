
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
    int y = (int)(uchar)x;
    int z = __ockl_clz_i32(y);
    return (char)(z - 24);
}

UEXPATTR uchar
clz(uchar x)
{
    int y = (int)x;
    int z = __ockl_clz_i32(y);
    return (char)(z - 24);
}

UEXPATTR short
clz(short x)
{
    int y = (int)(ushort)x;
    int z = __ockl_clz_i32(y);
    return (char)(z - 16);
}

UEXPATTR ushort
clz(ushort x)
{
    int y = (int)x;
    int z = __ockl_clz_i32(y);
    return (char)(z - 16);
}

UEXPATTR int
clz(int x)
{
    return __ockl_clz_i32(x);
}

UEXPATTR uint
clz(uint x)
{
    return __ockl_clz_i32((int)x);
}

__attribute__((always_inline, const)) static ulong
clz_u64(ulong x)
{
    int xlo = (int)x;
    int xhi = (int)(x >> 32);
    int zlo = __ockl_clz_i32(xlo) + 32;
    int zhi = __ockl_clz_i32(xhi);
    return (ulong)(xhi == 0 ? zlo : zhi);
}

extern __attribute__((overloadable, always_inline, const, alias("clz_u64"))) ulong clz(ulong);
extern __attribute__((overloadable, always_inline, const, alias("clz_u64")))  long clz(long);

