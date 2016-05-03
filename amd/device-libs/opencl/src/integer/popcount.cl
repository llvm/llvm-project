
#include "int.h"

#define UEXPATTR __attribute__((always_inline, overloadable, const))

UEXP(char,popcount)
UEXP(uchar,popcount)
UEXP(short,popcount)
UEXP(ushort,popcount)
UEXP(int,popcount)
UEXP(uint,popcount)
UEXP(long,popcount)
UEXP(ulong,popcount)

UEXPATTR char
popcount(char x)
{
    return (char)__llvm_ctpop_i32((int)(uchar)x);
}

UEXPATTR uchar
popcount(uchar x)
{
    return (uchar)__llvm_ctpop_i32((int)x);
}

UEXPATTR short
popcount(short x)
{
    return (short)__llvm_ctpop_i32((int)(ushort)x);
}

UEXPATTR ushort
popcount(ushort x)
{
    return (ushort)__llvm_ctpop_i32((int)x);
}

UEXPATTR int
popcount(int x)
{
    return __llvm_ctpop_i32(x);
}

UEXPATTR uint
popcount(uint x)
{
    return (uint)__llvm_ctpop_i32((int)x);
}

UEXPATTR long
popcount(long x)
{
    int2 y = as_int2(x);
    return (long)(__llvm_ctpop_i32(y.lo) + __llvm_ctpop_i32(y.hi));
}

UEXPATTR ulong
popcount(ulong x)
{
    int2 y = as_int2(x);
    return (ulong)(__llvm_ctpop_i32(y.lo) + __llvm_ctpop_i32(y.hi));
}

