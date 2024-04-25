/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "ockl.h"
#include "ockl_priv.h"

#define ATTR __attribute__((always_inline))

// Hack to prevent incorrect hoisting of the operation. There
// currently is no proper way in llvm to prevent hoisting of
// operations control flow dependent results.
ATTR
static int optimizationBarrierHack(int in_val)
{
    int out_val;
    __asm__ volatile ("" : "=v"(out_val) : "0"(in_val));
    return out_val;
}

ATTR bool
OCKL_MANGLE_I32(wfany)(int e)
{
    e = optimizationBarrierHack(e);
    return __builtin_amdgcn_ballot_w64(e) != 0;
}

ATTR bool
OCKL_MANGLE_I32(wfall)(int e)
{
    e = optimizationBarrierHack(e);
    return __builtin_amdgcn_ballot_w64(e) == __builtin_amdgcn_read_exec();
}

ATTR bool
OCKL_MANGLE_I32(wfsame)(int e)
{
    e = optimizationBarrierHack(e);
    ulong u = __builtin_amdgcn_ballot_w64(e);
    return (u == 0UL) | (u == __builtin_amdgcn_read_exec());
}

