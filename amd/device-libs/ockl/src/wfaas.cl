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

REQUIRES_WAVE32
static bool wfany_impl_w32(int e) {
    return __builtin_amdgcn_ballot_w32(e) != 0;
}

REQUIRES_WAVE64
static bool wfany_impl_w64(int e) {
    return __builtin_amdgcn_ballot_w64(e) != 0;
}

ATTR bool
OCKL_MANGLE_I32(wfany)(int e)
{
    e = optimizationBarrierHack(e);
    return __oclc_wavefrontsize64 ?
        wfany_impl_w64(e) : wfany_impl_w32(e);
}

REQUIRES_WAVE32
static bool wfall_impl_w32(int e) {
    return __builtin_amdgcn_ballot_w32(e) == __builtin_amdgcn_read_exec_lo();
}

REQUIRES_WAVE64
static bool wfall_impl_w64(int e) {
    return __builtin_amdgcn_ballot_w64(e) == __builtin_amdgcn_read_exec();
}

ATTR bool
OCKL_MANGLE_I32(wfall)(int e)
{
    e = optimizationBarrierHack(e);
    return __oclc_wavefrontsize64 ?
        wfall_impl_w64(e) : wfall_impl_w32(e);
}


REQUIRES_WAVE32
static bool wfsame_impl_w32(int e) {
    uint u = __builtin_amdgcn_ballot_w32(e);
    return (u == 0) | (u == __builtin_amdgcn_read_exec_lo());
}

REQUIRES_WAVE64
static bool wfsame_impl_w64(int e) {
    ulong u = __builtin_amdgcn_ballot_w64(e);
    return (u == 0UL) | (u == __builtin_amdgcn_read_exec());
}

ATTR bool
OCKL_MANGLE_I32(wfsame)(int e)
{
    e = optimizationBarrierHack(e);
    return __oclc_wavefrontsize64 ?
        wfsame_impl_w64(e) : wfsame_impl_w32(e);
}

