/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"

__attribute__((overloadable, always_inline)) void
sub_group_barrier(cl_mem_fence_flags flags)
{
    sub_group_barrier(flags, memory_scope_sub_group);
}

__attribute__((overloadable, always_inline)) void
sub_group_barrier(cl_mem_fence_flags flags, memory_scope scope)
{
    // This barrier is a no-op to ensure this function remains convergent
    __llvm_amdgcn_wave_barrier();

    if (flags)
        atomic_work_item_fence(flags, memory_order_acq_rel, scope);
}

