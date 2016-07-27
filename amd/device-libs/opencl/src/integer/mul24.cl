/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "int.h"

#define BEXPATTR __attribute__((always_inline, overloadable, const))

BEXP(int,mul24)
BEXP(uint,mul24)

BEXPATTR int
mul24(int x, int y)
{
    return __ockl_mul24_i32(x, y);
}

BEXPATTR uint
mul24(uint x, uint y)
{
    return __ockl_mul24_u32(x, y);
}

