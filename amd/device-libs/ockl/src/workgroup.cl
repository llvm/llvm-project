/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"

__attribute__((always_inline)) void
__ockl_work_group_barrier(cl_mem_fence_flags flags, memory_scope scope)
{
#if 0
    atomic_work_item_fence(flags, memory_order_release, scope);
    __llvm_amdgcn_s_barrier();
    atomic_work_item_fence(flags, memory_order_acquire, scope);
#else
    // Work around LIGHTNING-287
    __llvm_amdgcn_s_waitcnt(0);
    __llvm_amdgcn_s_dcache_wb();
    __llvm_amdgcn_s_barrier();
#endif
}

__attribute__((always_inline)) void
__ockl_barrier(cl_mem_fence_flags flags)
{
    __ockl_work_group_barrier(flags, memory_scope_work_group);
}
