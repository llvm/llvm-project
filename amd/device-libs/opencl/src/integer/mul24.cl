
#include "int.h"

#define BEXPATTR __attribute__((always_inline, overloadable, const))

BEXP(int,mul24)
BEXP(uint,mul24)

BEXPATTR int
mul24(int x, int y)
{
    return ((x << 8) >> 8) * ((y << 8) >> 8);
}

BEXPATTR uint
mul24(uint x, uint y)
{
    return ((x << 8) >> 8) * ((y << 8) >> 8);
}

