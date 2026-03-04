/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "ockl.h"

__attribute__((always_inline, const)) uchar
OCKL_MANGLE_T(clz,u8)(uchar i)
{
    return BUILTIN_CLZ_U8(i);
}

__attribute__((always_inline, const)) ushort
OCKL_MANGLE_T(clz,u16)(ushort i)
{
    return BUILTIN_CLZ_U16(i);
}

__attribute__((always_inline, const)) uint
OCKL_MANGLE_U32(clz)(uint i)
{
    return BUILTIN_CLZ_U32(i);
}

__attribute__((always_inline, const)) ulong
OCKL_MANGLE_U64(clz)(ulong i)
{
   return BUILTIN_CLZ_U64(i);
}

