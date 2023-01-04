/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"
__attribute__((const))
bool
OCKL_MANGLE_T(is_local,addr)(const void *a)
{
    return __builtin_amdgcn_is_shared(a);
}

__attribute__((const))
bool
OCKL_MANGLE_T(is_private,addr)(const void *a)
{
    return __builtin_amdgcn_is_private(a);
}

__attribute__((const)) __global void *
OCKL_MANGLE_T(to,global)(void *a)
{
    return (OCKL_MANGLE_T(is_local,addr)(a) |
            OCKL_MANGLE_T(is_private,addr)(a)) ?
        (__global void *)0 : (__global void*)a;
}

__attribute__((const)) __local void *
OCKL_MANGLE_T(to,local)(void *a)
{
    return OCKL_MANGLE_T(is_local,addr)(a) ?
        (__local void *)a : (__local void *)0;
}

__attribute__((const)) __private void *
OCKL_MANGLE_T(to,private)(void *a)
{
    return OCKL_MANGLE_T(is_private,addr)(a) ?
        (__private void *)a : (__private void *)0;
}

