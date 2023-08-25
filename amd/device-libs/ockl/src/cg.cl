/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "ockl.h"

#define AL(P,S) __opencl_atomic_load((__global atomic_uint *)P, memory_order_relaxed, S)
#define AA(P,V,S) __opencl_atomic_fetch_add((__global atomic_uint *)P, V, memory_order_relaxed, S)

#define AVOID_GWS() (__oclc_ISA_version == 9400 || __oclc_ISA_version == 9401 || __oclc_ISA_version == 9402 || __oclc_ISA_version >= 11000)

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

    struct mg_sync sgs;
    uint num_wg;
};

static inline size_t
get_mg_info_arg(void)
{
    if (__oclc_ABI_version < 500) {
        return ((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[6];
    } else {
        return ((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[11];
    }
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
single_grid_sync(__global struct mg_sync *s, uint members)
{
    // Assumes 65535 or fewer workgroups in the grid
    uint v = AA(&s->w0, 1U, memory_scope_device);
    if ((v & 0xffff) == members-1) {
        AA(&s->w0, 0x10000 - members, memory_scope_device);
    } else {
        v &= ~0xffff;
        do {
            __builtin_amdgcn_s_sleep(1);
        } while ((AL(&s->w0, memory_scope_device) & ~0xffff) == v);
    }
}

static inline void
multi_grid_sync(__global struct mg_sync *s, uint members)
{
    // Assumes 255 or fewer GPUs in the multi grid
    uint v = AA(&s->w0, 1U, memory_scope_all_svm_devices);
    if ((v & 0xff) == members-1) {
        AA(&s->w0, 0x100 - members, memory_scope_all_svm_devices);
    } else {
        v &= ~0xff;
        do {
            __builtin_amdgcn_s_sleep(2);
        } while ((AL(&s->w0, memory_scope_all_svm_devices) & ~0xff) == v);
    }
}

__attribute__((target("gws"))) void
__ockl_gws_init(uint nwm1, uint rid)
{
    __builtin_amdgcn_ds_gws_init(nwm1, rid);
}

__attribute__((target("gws"))) void
__ockl_gws_barrier(uint nwm1, uint rid)
{
    __builtin_amdgcn_ds_gws_barrier(nwm1, rid);
}

__attribute__((const)) int
__ockl_grid_is_valid(void)
{
    return get_mg_info_arg() != 0UL;
}

void
__ockl_grid_sync(void)
{
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
    __builtin_amdgcn_s_barrier();
    if (choose_one_workgroup_workitem()) {
        if (AVOID_GWS()) {
            __global struct mg_info *mi = (__global struct mg_info *)get_mg_info_arg();
            single_grid_sync(&mi->sgs, mi->num_wg);
        } else {
            uint nwm1 = (uint)__ockl_get_num_groups(0) * (uint)__ockl_get_num_groups(1) * (uint)__ockl_get_num_groups(2) - 1;
            __ockl_gws_barrier(nwm1, 0);
        }
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
    if (AVOID_GWS()) {
         __constant struct mg_info *mi = (__constant struct mg_info *)get_mg_info_arg();
        return mi && mi->num_grids > 0;
    } else {
        return get_mg_info_arg() > 1;
    }
}

void
__ockl_multi_grid_sync(void)
{
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
    __builtin_amdgcn_s_barrier();

    bool cwwi = choose_one_workgroup_workitem();
    uint nwm1 = (uint)__ockl_get_num_groups(0) * (uint)__ockl_get_num_groups(1) * (uint)__ockl_get_num_groups(2) - 1;
    __global struct mg_info *mi = (global struct mg_info *)get_mg_info_arg();
    uint nwg = mi->num_wg;
    __global struct mg_sync *sgs = &mi->sgs;

    if (cwwi) {
        if (AVOID_GWS()) {
            single_grid_sync(sgs, nwg);
        } else {
            __ockl_gws_barrier(nwm1, 0);
        }
    }

    if (choose_one_grid_workitem()) {
        multi_grid_sync(mi->mgs, mi->num_grids);
    }

    if (cwwi) {
        if (AVOID_GWS()) {
            single_grid_sync(sgs, nwg);
        } else {
            __ockl_gws_barrier(nwm1, 0);
        }
    }

    __builtin_amdgcn_s_barrier();
}

