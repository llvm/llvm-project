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

// Hack to prevent incorrect hoisting of the operation. There
// currently is no proper way in llvm to prevent hoisting of
// operations control flow dependent results.
ATTR
static int optimizationBarrierHack(int in_val)
{
    int out_val;
    __asm__ volatile ("; ockl ballot hoisting hack %0" :
                      "=v"(out_val) : "0"(in_val));
    return out_val;
}

ATTR bool
OCKL_MANGLE_I32(wfany)(int e)
{
    e = optimizationBarrierHack(e);
    return __builtin_amdgcn_sicmp(e, 0, ICMP_NE) != 0UL;
}

ATTR bool
OCKL_MANGLE_I32(wfall)(int e)
{
    e = optimizationBarrierHack(e);
    return __builtin_amdgcn_sicmp(e, 0, ICMP_NE) == __builtin_amdgcn_read_exec();
}


ATTR bool
OCKL_MANGLE_I32(wfsame)(int e)
{
    e = optimizationBarrierHack(e);
    ulong u = __builtin_amdgcn_sicmp(e, 0, ICMP_NE) != 0;
    return (u == 0UL) | (u == __builtin_amdgcn_read_exec());
}

