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
    __builtin_amdgcn_ds_gws_init(nwm1, rid);
}

__attribute__((convergent)) void
__ockl_gws_barrier(uint nwm1, uint rid)
{
    __builtin_amdgcn_ds_gws_barrier(nwm1, rid);
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

