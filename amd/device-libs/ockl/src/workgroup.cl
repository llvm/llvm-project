/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"

__attribute__((always_inline)) void
__ockl_barrier(cl_mem_fence_flags flags)
{
#if 0
    work_group_barrier(flags);
#else
    // Work around LIGHTNING-287
    __llvm_amdgcn_s_waitcnt(0);
    __llvm_amdgcn_s_dcache_wb();
    __llvm_amdgcn_s_barrier();
#endif
}
