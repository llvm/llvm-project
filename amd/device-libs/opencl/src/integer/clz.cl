
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
    int z = y ? __llvm_ctlz_i32(y) : 32;
    return (char)(z - 24);
}

UEXPATTR uchar
clz(uchar x)
{
    int y = (int)x;
    int z = y ? __llvm_ctlz_i32(y) : 32;
    return (char)(z - 24);
}

UEXPATTR short
clz(short x)
{
    int y = (int)(ushort)x;
    int z = y ? __llvm_ctlz_i32(y) : 32;
    return (char)(z - 16);
}

UEXPATTR ushort
clz(ushort x)
{
    int y = (int)x;
    int z = y ? __llvm_ctlz_i32(y) : 32;
    return (char)(z - 16);
}

UEXPATTR int
clz(int x)
{
    return x ? __llvm_ctlz_i32(x) : 32;
}

UEXPATTR uint
clz(uint x)
{
    return x ? __llvm_ctlz_i32((int)x) : 32;
}

__attribute__((always_inline, const)) static ulong
clz_u64(ulong x)
{
    int xlo = (int)x;
    int xhi = (int)(x >> 32);
    int zlo = (xlo ? __llvm_ctlz_i32(xlo) : 32) + 32;
    int zhi = xhi ? __llvm_ctlz_i32(xlo) : 32;
    return (ulong)(xhi == 0 ? zlo : zhi);
}

extern __attribute__((overloadable, always_inline, const, alias("clz_u64"))) ulong clz(ulong);
extern __attribute__((overloadable, always_inline, const, alias("clz_u64")))  long clz(long);

