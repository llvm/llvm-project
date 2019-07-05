/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "ockl.h"

__attribute__((convergent)) void
__ockl_gws_init(uint nwm1, uint rid)
{
    // TODO replace with intrinsic call when working
    uint m0_backup, new_m0;
    __asm__ __volatile__(
           " s_mov_b32 %0 m0\n"
           " v_readfirstlane_b32 %1 %2\n"
           " s_nop 0\n"
           " s_mov_b32 m0 %1\n"
           " s_nop 0\n"
           " ds_gws_init %3 offset:0 gds\n"
           " s_waitcnt 0\n"
           " s_mov_b32 m0 %0\n"
           " s_nop 0"
           : "=s"(m0_backup), "=s"(new_m0)
           : "v"(rid<<0x10), "{v0}"(nwm1)
           : "memory");
}

__attribute__((convergent)) void
__ockl_gws_barrier(uint nwm1, uint rid)
{
    // TODO replace with intrinsic call when working
    uint m0_backup, new_m0;
    __asm__ __volatile__(
        " s_mov_b32 %0 m0\n"
        " v_readfirstlane_b32 %1 %2\n"
        " s_nop 0\n"
        " s_mov_b32 m0 %1\n"
        " s_nop 0\n"
        " ds_gws_barrier %3 offset:0 gds\n"
        " s_waitcnt 0\n"
        " s_mov_b32 m0 %0\n"
        " s_nop 0"
        : "=s"(m0_backup), "=s"(new_m0)
        : "v"(rid << 0x10), "{v0}"(nwm1)
        : "memory");
}

__attribute__((convergent)) void
__ockl_grid_sync(void)
{
    __llvm_fence_sc_dev();
    if (__ockl_get_local_linear_id() == 0) {
        uint nwm1 = (uint)__ockl_get_num_groups(0) * (uint)__ockl_get_num_groups(1) * (uint)__ockl_get_num_groups(2) - 1;
        __ockl_gws_barrier(nwm1, 0);
    }
    __builtin_amdgcn_s_barrier();
}

