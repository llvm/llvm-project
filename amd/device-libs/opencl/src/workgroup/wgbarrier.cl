/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

__attribute__((overloadable)) void
barrier(cl_mem_fence_flags flags)
{
    work_group_barrier(flags);
}

__attribute__((overloadable)) void
work_group_barrier(cl_mem_fence_flags flags)
{
    work_group_barrier(flags, memory_scope_work_group);
}

__attribute__((overloadable)) void
work_group_barrier(cl_mem_fence_flags flags, memory_scope scope)
{
    if (flags) {
        atomic_work_item_fence(flags,
            flags == (CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE) ?
                memory_order_seq_cst : memory_order_release,
            scope);

        __builtin_amdgcn_s_barrier();

        atomic_work_item_fence(flags,
            flags == (CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE) ?
                memory_order_seq_cst : memory_order_acquire,
            scope);
    } else {
        __builtin_amdgcn_s_barrier();
    }
}

