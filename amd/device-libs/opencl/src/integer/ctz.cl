
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
    return (char)(y ? __llvm_cttz_i32(y) : 8);
}

UEXPATTR uchar
ctz(uchar x)
{
    int y = (int)x;
    return (uchar)(y ? __llvm_cttz_i32(y) : 8);
}

UEXPATTR short
ctz(short x)
{
    int y = (int)(ushort)x;
    return (short)(y ? __llvm_cttz_i32(y) : 16);
}

UEXPATTR ushort
ctz(ushort x)
{
    int y = (int)x;
    return (ushort)(y ? __llvm_cttz_i32(y) : 16);
}

UEXPATTR int
ctz(int x)
{
    return x ? __llvm_cttz_i32(x) : 32;
}

UEXPATTR uint
ctz(uint x)
{
    return x ? (uint)__llvm_cttz_i32((int)x) : 32u;
}

__attribute__((always_inline, const)) static ulong
ctz_u64(ulong x)
{
    int xlo = (int)x;
    int xhi = (int)(x >> 32);
    int zlo = xlo ? __llvm_cttz_i32(xlo) : 32;
    int zhi = (xhi ? __llvm_cttz_i32(xlo) : 32) + 32;
    return (ulong)(xlo == 0 ? zhi : zlo);
}

extern __attribute__((overloadable, always_inline, const, alias("ctz_u64"))) ulong ctz(ulong);
extern __attribute__((overloadable, always_inline, const, alias("ctz_u64")))  long ctz(long);

