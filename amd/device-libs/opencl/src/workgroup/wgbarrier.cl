/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

__attribute__((overloadable, always_inline)) void
barrier(cl_mem_fence_flags flags)
{
    __ockl_barrier(flags);
}

__attribute__((overloadable, always_inline)) void
work_group_barrier(cl_mem_fence_flags flags)
{
    work_group_barrier(flags, memory_scope_work_group);
}

__attribute__((overloadable, always_inline)) void
work_group_barrier(cl_mem_fence_flags flags, memory_scope scope)
{
    __ockl_work_group_barrier(flags, scope);
}

