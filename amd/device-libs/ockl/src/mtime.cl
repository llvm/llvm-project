/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

__attribute__((target("s-memtime-inst"))) ulong
OCKL_MANGLE_U64(memtime)(void)
{
    return __builtin_amdgcn_s_memtime();
}

__attribute__((target("s-memrealtime"))) ulong
OCKL_MANGLE_U64(memrealtime)(void)
{
    return __builtin_amdgcn_s_memrealtime();
}

