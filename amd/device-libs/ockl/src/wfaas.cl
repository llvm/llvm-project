/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "ockl.h"

#define ATTR __attribute__((always_inline))

ATTR bool
OCKL_MANGLE_I32(wfany)(int e)
{
    return __builtin_amdgcn_ballot_w64(e) != 0;
}

ATTR bool
OCKL_MANGLE_I32(wfall)(int e)
{
    return __builtin_amdgcn_ballot_w64(e) == __builtin_amdgcn_read_exec();
}

ATTR bool
OCKL_MANGLE_I32(wfsame)(int e)
{
    ulong u = __builtin_amdgcn_ballot_w64(e);
    return (u == 0UL) | (u == __builtin_amdgcn_read_exec());
}

