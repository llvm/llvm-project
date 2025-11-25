/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

__attribute__((always_inline, const)) uint
OCKL_MANGLE_U32(popcount)(uint i)
{
    return (uint)__builtin_popcount(i);
}

__attribute__((always_inline, const)) ulong
OCKL_MANGLE_U64(popcount)(ulong i)
{
    return (ulong)__builtin_popcountl(i);
}

