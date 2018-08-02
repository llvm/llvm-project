/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "ockl.h"

#define ATTR __attribute__((always_inline))

// XXX from llvm/include/llvm/IR/InstrTypes.h
#define ICMP_NE 33

ATTR bool
OCKL_MANGLE_I32(wfany)(int e)
{
    return __builtin_amdgcn_sicmp(e, 0, ICMP_NE) != 0UL;
}

ATTR bool
OCKL_MANGLE_I32(wfall)(int e)
{
    return __builtin_amdgcn_sicmp(e, 0, ICMP_NE) == __builtin_amdgcn_read_exec();
}


ATTR bool
OCKL_MANGLE_I32(wfsame)(int e)
{
    ulong u = __builtin_amdgcn_sicmp(e, 0, ICMP_NE) != 0;
    return (u == 0UL) | (u == __builtin_amdgcn_read_exec());
}

