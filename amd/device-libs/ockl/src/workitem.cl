/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "device_amd_hsa.h"

#define ATTR __attribute__((const))
#define OLD_ABI __oclc_ABI_version < 500

#define IMPLICITARG(T) ((__constant T *)__builtin_amdgcn_implicitarg_ptr())

ATTR static size_t
get_global_offset_x(void)
{
    if (OLD_ABI) {
        return IMPLICITARG(ulong)[0];
    } else {
        return IMPLICITARG(ulong)[5];
    }
}

ATTR static size_t
get_global_offset_y(void)
{
    if (OLD_ABI) {
        return IMPLICITARG(ulong)[1];
    } else {
        return IMPLICITARG(ulong)[6];
    }
}

ATTR static size_t
get_global_offset_z(void)
{
    if (OLD_ABI) {
        return IMPLICITARG(ulong)[2];
    } else {
        return IMPLICITARG(ulong)[7];
    }
}

ATTR static size_t
get_global_size_x(void)
{
    if (OLD_ABI) {
        __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();
        return p->grid_size_x;
    } else {
        return IMPLICITARG(uint)[0]*IMPLICITARG(ushort)[6] + IMPLICITARG(ushort)[9];
        return 0;
    }
}

ATTR static size_t
get_global_size_y(void)
{
    if (OLD_ABI) {
        __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();
        return p->grid_size_y;
    } else {
        return IMPLICITARG(uint)[1]*IMPLICITARG(ushort)[7] + IMPLICITARG(ushort)[10];
    }
}

ATTR static size_t
get_global_size_z(void)
{
    if (OLD_ABI) {
        __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();
        return p->grid_size_z;
    } else {
        return IMPLICITARG(uint)[2]*IMPLICITARG(ushort)[8] + IMPLICITARG(ushort)[11];
        return 0;
    }
}

ATTR static size_t
get_global_id_x(void)
{
    uint l = __builtin_amdgcn_workitem_id_x();
    uint g = __builtin_amdgcn_workgroup_id_x();
    uint s;
    if (OLD_ABI) {
        s = __builtin_amdgcn_workgroup_size_x();
    } else {
        s = IMPLICITARG(ushort)[6];
    }
    return (g*s + l) + get_global_offset_x();
}

ATTR static size_t
get_global_id_y(void)
{
    uint l = __builtin_amdgcn_workitem_id_y();
    uint g = __builtin_amdgcn_workgroup_id_y();
    uint s;
    if (OLD_ABI) {
        s = __builtin_amdgcn_workgroup_size_y();
    } else {
        s = IMPLICITARG(ushort)[7];
    }
    return (g*s + l) + get_global_offset_y();
}

ATTR static size_t
get_global_id_z(void)
{
    uint l = __builtin_amdgcn_workitem_id_z();
    uint g = __builtin_amdgcn_workgroup_id_z();
    uint s;
    if (OLD_ABI) {
        s = __builtin_amdgcn_workgroup_size_z();
    } else {
        s = IMPLICITARG(ushort)[8];
    }
    return (g*s + l) + get_global_offset_z();
}

ATTR static size_t
get_local_size_x(void)
{
    if (OLD_ABI) {
        __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();
        uint group_id = __builtin_amdgcn_workgroup_id_x();
        uint group_size = __builtin_amdgcn_workgroup_size_x();
        uint grid_size = p->grid_size_x;
        uint r = grid_size - group_id * group_size;
        return (r < group_size) ? r : group_size;
    } else {
        return __builtin_amdgcn_workgroup_id_x() < IMPLICITARG(uint)[0] ? IMPLICITARG(ushort)[6] : IMPLICITARG(ushort)[9];
    }
}

ATTR static size_t
get_local_size_y(void)
{
    if (OLD_ABI) {
        __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();
        uint group_id = __builtin_amdgcn_workgroup_id_y();
        uint group_size = __builtin_amdgcn_workgroup_size_y();
        uint grid_size = p->grid_size_y;
        uint r = grid_size - group_id * group_size;
        return (r < group_size) ? r : group_size;
    } else {
        return __builtin_amdgcn_workgroup_id_y() < IMPLICITARG(uint)[1] ? IMPLICITARG(ushort)[7] : IMPLICITARG(ushort)[10];
    }
}

ATTR static size_t
get_local_size_z(void)
{
    if (OLD_ABI) {
        __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();
        uint group_id = __builtin_amdgcn_workgroup_id_z();
        uint group_size = __builtin_amdgcn_workgroup_size_z();
        uint grid_size = p->grid_size_z;
        uint r = grid_size - group_id * group_size;
        return (r < group_size) ? r : group_size;
    } else {
        return __builtin_amdgcn_workgroup_id_z() < IMPLICITARG(uint)[2] ? IMPLICITARG(ushort)[8] : IMPLICITARG(ushort)[11];
    }
}

ATTR static size_t
get_enqueued_local_size_x(void)
{
    if (OLD_ABI) {
        return __builtin_amdgcn_workgroup_size_x();
    } else {
        return IMPLICITARG(ushort)[6];
    }
}

ATTR static size_t
get_enqueued_local_size_y(void)
{
    if (OLD_ABI) {
        return __builtin_amdgcn_workgroup_size_y();
    } else {
        return IMPLICITARG(ushort)[7];
    }
}

ATTR static size_t
get_enqueued_local_size_z(void)
{
    if (OLD_ABI) {
        return __builtin_amdgcn_workgroup_size_z();
    } else {
        return IMPLICITARG(ushort)[8];
    }
}

ATTR static size_t
get_num_groups_x(void)
{
    if (OLD_ABI) {
        __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();
        uint n = p->grid_size_x;
        uint d = __builtin_amdgcn_workgroup_size_x();
        uint q = n / d;
        return q + (n > q*d);
    } else {
        return IMPLICITARG(uint)[0] + (IMPLICITARG(ushort)[9] > 0);
    }
}

ATTR static size_t
get_num_groups_y(void)
{
    if (OLD_ABI) {
        __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();
        uint n = p->grid_size_y;
        uint d = __builtin_amdgcn_workgroup_size_y();
        uint q = n / d;
        return q + (n > q*d);
    } else {
        return IMPLICITARG(uint)[1] + (IMPLICITARG(ushort)[10] > 0);
    }
}

ATTR static size_t
get_num_groups_z(void)
{
    if (OLD_ABI) {
        __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();
        uint n = p->grid_size_z;
        uint d = __builtin_amdgcn_workgroup_size_z();
        uint q = n / d;
        return q + (n > q*d);
    } else {
        return IMPLICITARG(uint)[2] + (IMPLICITARG(ushort)[11] > 0);
    }
}

ATTR static uint
get_work_dim_(void)
{
    if (OLD_ABI) {
        __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();
        return p->setup;
    } else {
        return IMPLICITARG(ushort)[32];
    }
}

ATTR static size_t
get_global_linear_id_x(void)
{
    uint l0 = __builtin_amdgcn_workitem_id_x();
    uint g0 = __builtin_amdgcn_workgroup_id_x();
    uint s0;
    if (OLD_ABI) {
        s0 = __builtin_amdgcn_workgroup_size_x();
    } else {
        s0 = IMPLICITARG(ushort)[6];
    }
    return g0*s0 + l0;
}

ATTR static size_t
get_global_linear_id_y(void)
{
    uint l0 = __builtin_amdgcn_workitem_id_x();
    uint l1 = __builtin_amdgcn_workitem_id_y();
    uint g0 = __builtin_amdgcn_workgroup_id_x();
    uint g1 = __builtin_amdgcn_workgroup_id_y();
    uint s0, s1;
    uint n0;

    if (OLD_ABI) {
        __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();
        s0 = __builtin_amdgcn_workgroup_size_x();
        s1 = __builtin_amdgcn_workgroup_size_y();
        n0 = p->grid_size_x;
    } else {
        s0 = IMPLICITARG(ushort)[6];
        s1 = IMPLICITARG(ushort)[7];
        n0 = IMPLICITARG(uint)[0]*s0 + IMPLICITARG(ushort)[9];
    }
    uint i0 = g0*s0 + l0;
    uint i1 = g1*s1 + l1;
    return (size_t)i1 * (size_t)n0 + i0;
}

ATTR static size_t
get_global_linear_id_z(void)
{
    uint l0 = __builtin_amdgcn_workitem_id_x();
    uint l1 = __builtin_amdgcn_workitem_id_y();
    uint l2 = __builtin_amdgcn_workitem_id_z();
    uint g0 = __builtin_amdgcn_workgroup_id_x();
    uint g1 = __builtin_amdgcn_workgroup_id_y();
    uint g2 = __builtin_amdgcn_workgroup_id_z();
    uint s0, s1, s2;
    uint n0, n1;

    if (OLD_ABI) {
        __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();
        s0 = __builtin_amdgcn_workgroup_size_x();
        s1 = __builtin_amdgcn_workgroup_size_y();
        s2 = __builtin_amdgcn_workgroup_size_z();
        n0 = p->grid_size_x;
        n1 = p->grid_size_y;
    } else {
        s0 = IMPLICITARG(ushort)[6];
        s1 = IMPLICITARG(ushort)[7];
        s2 = IMPLICITARG(ushort)[8];
        n0 = IMPLICITARG(uint)[0]*s0 + IMPLICITARG(ushort)[9];
        n1 = IMPLICITARG(uint)[1]*s1 + IMPLICITARG(ushort)[10];
    }
    uint i0 = g0*s0 + l0;
    uint i1 = g1*s1 + l1;
    uint i2 = g2*s2 + l2;
    return ((size_t)i2 * (size_t)n1 + (size_t)i1) * (size_t)n0 + i0;
}

ATTR static size_t
get_local_linear_id_(void)
{
    return (__builtin_amdgcn_workitem_id_z()  * (uint)get_local_size_y() +
            __builtin_amdgcn_workitem_id_y()) * (uint)get_local_size_x() +
            __builtin_amdgcn_workitem_id_x();
}

ATTR size_t
__ockl_get_global_offset(uint dim)
{
    switch(dim) {
    case 0:
        return get_global_offset_x();
    case 1:
        return get_global_offset_y();
    case 2:
        return get_global_offset_z();
    default:
        return 0;
    }
}

ATTR size_t
__ockl_get_global_id(uint dim)
{
    switch(dim) {
    case 0:
        return get_global_id_x();
    case 1:
        return get_global_id_y();
    case 2:
        return get_global_id_z();
    default:
        return 0;
    }
}

ATTR size_t
__ockl_get_local_id(uint dim)
{
    switch(dim) {
    case 0:
        return __builtin_amdgcn_workitem_id_x();
    case 1:
        return __builtin_amdgcn_workitem_id_y();
    case 2:
        return __builtin_amdgcn_workitem_id_z();
    default:
        return 0;
    }
}

ATTR size_t
__ockl_get_group_id(uint dim)
{
    switch(dim) {
    case 0:
        return __builtin_amdgcn_workgroup_id_x();
    case 1:
        return __builtin_amdgcn_workgroup_id_y();
    case 2:
        return __builtin_amdgcn_workgroup_id_z();
    default:
        return 0;
    }
}

ATTR size_t
__ockl_get_global_size(uint dim)
{
    switch(dim) {
    case 0:
        return get_global_size_x();
    case 1:
        return get_global_size_y();
    case 2:
        return get_global_size_z();
    default:
        return 1;
    }
}

ATTR size_t
__ockl_get_local_size(uint dim)
{
    switch(dim) {
    case 0:
        return get_local_size_x();
    case 1:
        return get_local_size_y();
    case 2:
        return get_local_size_z();
    default:
        return 1;
    }
}

ATTR size_t
__ockl_get_num_groups(uint dim)
{
    switch(dim) {
    case 0:
        return get_num_groups_x();
    case 1:
        return get_num_groups_y();
    case 2:
        return get_num_groups_z();
    default:
        return 1;
    }
}

ATTR uint
__ockl_get_work_dim(void)
{
    return get_work_dim_();
}

ATTR size_t
__ockl_get_enqueued_local_size(uint dim)
{
    switch(dim) {
    case 0:
        return get_enqueued_local_size_x();
    case 1:
        return get_enqueued_local_size_y();
    case 2:
        return get_enqueued_local_size_z();
    default:
        return 1;
    }
}

ATTR size_t
__ockl_get_global_linear_id(void)
{
    switch (get_work_dim_()) {
    case 1:
        return get_global_linear_id_x();
    case 2:
        return get_global_linear_id_y();
    case 3:
        return get_global_linear_id_z();
    default:
        return 0;
    }
}

ATTR size_t
__ockl_get_local_linear_id(void)
{
    return get_local_linear_id_();
}

