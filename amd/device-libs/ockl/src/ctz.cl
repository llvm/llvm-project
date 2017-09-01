/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "ockl.h"

__attribute__((always_inline, const)) uchar
OCKL_MANGLE_T(ctz,u8)(uchar i)
{
    return __llvm_cttz_i8(i);
}

__attribute__((always_inline, const)) ushort
OCKL_MANGLE_T(ctz,u16)(ushort i)
{
    return __llvm_cttz_i16(i);
}

__attribute__((always_inline, const)) uint
OCKL_MANGLE_U32(ctz)(uint i)
{
    return __llvm_cttz_i32(i);
}

__attribute__((always_inline, const)) ulong
OCKL_MANGLE_U64(ctz)(ulong i)
{
    return __llvm_cttz_i64(i);
}

