/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "ockl.h"

#define AL(P) __opencl_atomic_load((__global atomic_uint *)P, memory_order_relaxed, memory_scope_all_svm_devices)
#define AA(P,V) __opencl_atomic_fetch_add((__global atomic_uint *)P, V, memory_order_relaxed, memory_scope_all_svm_devices)

// XXX do not change these two structs without changing the language runtime
struct mg_sync {
    uint w0;
    uint w1;
};

struct mg_info {
    __global struct mg_sync *mgs;
    uint grid_id;
    uint num_grids;
    ulong prev_sum;
    ulong all_sum;
};

static inline size_t
get_mg_info_arg(void)
{
    return ((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[6];
}

static inline bool
choose_one_workgroup_workitem(void)
{
    return (__builtin_amdgcn_workitem_id_x() | __builtin_amdgcn_workitem_id_y() | __builtin_amdgcn_workitem_id_z()) == 0;
}

static inline bool
choose_one_grid_workitem(void)
{
    return (__builtin_amdgcn_workitem_id_x() | __builtin_amdgcn_workgroup_id_x() |
            __builtin_amdgcn_workitem_id_y() | __builtin_amdgcn_workgroup_id_y() |
            __builtin_amdgcn_workitem_id_z() | __builtin_amdgcn_workgroup_id_z()) == 0;
}

static inline void
multi_grid_sync(__global struct mg_sync *s, uint members)
{
    // Assumes 255 or fewer GPUs in multi_grid
    uint v = AA(&s->w0, 1U);
    if ((v & 0xff) == members-1) {
        AA(&s->w0, 0x100 - members);
    } else {
        v &= ~0xff;
        do {
            __builtin_amdgcn_s_sleep(2);
        } while ((AL(&s->w0) & ~0xff) == v);
    }
}

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

__attribute__((const)) int
__ockl_grid_is_valid(void)
{
    return get_mg_info_arg() != 0UL;
}

__attribute__((convergent)) void
__ockl_grid_sync(void)
{
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
    if (choose_one_workgroup_workitem()) {
        uint nwm1 = (uint)__ockl_get_num_groups(0) * (uint)__ockl_get_num_groups(1) * (uint)__ockl_get_num_groups(2) - 1;
        __ockl_gws_barrier(nwm1, 0);
    }
    __builtin_amdgcn_s_barrier();
}

__attribute__((const)) uint
__ockl_multi_grid_num_grids(void)
{
    return ((__constant struct mg_info *)get_mg_info_arg())->num_grids;
}

__attribute__((const)) uint
__ockl_multi_grid_grid_rank(void)
{
    return ((__constant struct mg_info *)get_mg_info_arg())->grid_id;
}

__attribute__((const)) uint
__ockl_multi_grid_size(void)
{
    return ((__constant struct mg_info *)get_mg_info_arg())->all_sum;
}

__attribute__((const)) uint
__ockl_multi_grid_thread_rank(void)
{
    size_t r = ((__constant struct mg_info *)get_mg_info_arg())->prev_sum;
    r += __ockl_get_global_linear_id();
    return r;
}

__attribute__((const)) int
__ockl_multi_grid_is_valid(void)
{
    size_t mi = get_mg_info_arg();
    return (mi != 0UL) & (mi != 1UL);
}

__attribute__((convergent)) void
__ockl_multi_grid_sync(void)
{
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
    uint nwm1 = (uint)__ockl_get_num_groups(0) * (uint)__ockl_get_num_groups(1) * (uint)__ockl_get_num_groups(2) - 1;
    bool cwwi = choose_one_workgroup_workitem();

    if (cwwi)
        __ockl_gws_barrier(nwm1, 0);

    __builtin_amdgcn_s_barrier();

    if (choose_one_grid_workitem()) {
        __constant struct mg_info *m = (__constant struct mg_info *)get_mg_info_arg();
        multi_grid_sync(m->mgs, m->num_grids);
    }

    if (cwwi)
        __ockl_gws_barrier(nwm1, 0);

    __builtin_amdgcn_s_barrier();
}

