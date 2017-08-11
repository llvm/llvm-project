/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "ockl.h"

__attribute__((always_inline, const)) uint
OCKL_MANGLE_U32(ctz)(uint i)
{
    uint r = (uint)__llvm_cttz_i32((int)i);
    return i ? r : 32u;
}

__attribute__((always_inline, const)) ulong
OCKL_MANGLE_U64(ctz)(ulong i)
{
    ulong r = (ulong)__llvm_cttz_i64((long)i);
    return i ? r : 64ul;
}

