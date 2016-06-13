
#include "ockl.h"

__attribute__((always_inline, const)) int
OCKL_MANGLE_I32(mul24)(int x, int y)
{
    return ((x << 8) >> 8) * ((y << 8) >> 8);
}

__attribute__((always_inline, const)) uint
OCKL_MANGLE_U32(mul24)(uint x, uint y)
{
    return ((x << 8) >> 8) * ((y << 8) >> 8);
}

