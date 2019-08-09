/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define GEN(A) \
__attribute__((always_inline, overloadable)) void \
wait_group_events(int n, A event_t *evs) \
{ \
    work_group_barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE, memory_scope_work_group); \
}

GEN(__private)
GEN()

