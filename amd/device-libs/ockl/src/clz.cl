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
    return __llvm_ctlz_i8(i);
}

__attribute__((always_inline, const)) ushort
OCKL_MANGLE_T(clz,u16)(ushort i)
{
    return __llvm_ctlz_i16(i);
}

__attribute__((always_inline, const)) uint
OCKL_MANGLE_U32(clz)(uint i)
{
    return __llvm_ctlz_i32(i);
}

__attribute__((always_inline, const)) ulong
OCKL_MANGLE_U64(clz)(ulong i)
{
   return __llvm_ctlz_i64(i);
}

