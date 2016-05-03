
#include "int.h"

#define TEXPATTR __attribute__((always_inline, overloadable, const))

TEXP(int,mad24)
TEXP(uint,mad24)

TEXPATTR int
mad24(int a, int b, int c)
{
    return ((a << 8) >> 8) * ((b << 8) >> 8) + c;
}

TEXPATTR uint
mad24(uint a, uint b, uint c)
{
    return (a & 0xffffffU) * (b & 0xffffffU) + c;
}

