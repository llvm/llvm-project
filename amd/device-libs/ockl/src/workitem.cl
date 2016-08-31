/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "device_amd_hsa.h"

#define ATTR __attribute__((always_inline, const))

ATTR size_t
__ockl_get_global_offset(uint dim)
{
    // TODO find out if implicit arg pointer is aligned properly
    switch(dim) {
    case 0:
        return *(__constant size_t *)__llvm_amdgcn_implicitarg_ptr();
    case 1:
        return ((__constant size_t *)__llvm_amdgcn_implicitarg_ptr())[1];
    case 2:
        return ((__constant size_t *)__llvm_amdgcn_implicitarg_ptr())[2];
    default:
        return 0;
    }
}

ATTR size_t
__ockl_get_global_id(uint dim)
{
    __constant hsa_kernel_dispatch_packet_t *p = __llvm_amdgcn_dispatch_ptr();
    uint l, g, s;

    switch(dim) {
    case 0:
        l = __llvm_amdgcn_workitem_id_x();
        g = __llvm_amdgcn_workgroup_id_x();
        s = p->workgroup_size_x;
        break;
    case 1:
        l = __llvm_amdgcn_workitem_id_y();
        g = __llvm_amdgcn_workgroup_id_y();
        s = p->workgroup_size_y;
        break;
    case 2:
        l = __llvm_amdgcn_workitem_id_z();
        g = __llvm_amdgcn_workgroup_id_z();
        s = p->workgroup_size_z;
        break;
    default:
        l = 0;
        g = 0;
        s = 1;
        break;
    }

    return (g*s + l) + __ockl_get_global_offset(dim);
}

ATTR size_t
__ockl_get_local_id(uint dim)
{
    switch(dim) {
    case 0:
        return __llvm_amdgcn_workitem_id_x();
    case 1:
        return __llvm_amdgcn_workitem_id_y();
    case 2:
        return __llvm_amdgcn_workitem_id_z();
    default:
        return 0;
    }
}

ATTR size_t
__ockl_get_group_id(uint dim)
{
    switch(dim) {
    case 0:
        return __llvm_amdgcn_workgroup_id_x();
    case 1:
        return __llvm_amdgcn_workgroup_id_y();
    case 2:
        return __llvm_amdgcn_workgroup_id_z();
    default:
        return 0;
    }
}

ATTR size_t
__ockl_get_global_size(uint dim)
{
    __constant hsa_kernel_dispatch_packet_t *p = __llvm_amdgcn_dispatch_ptr();

    switch(dim) {
    case 0:
        return p->grid_size_x;
    case 1:
        return p->grid_size_y;
    case 2:
        return p->grid_size_z;
    default:
        return 1;
    }
}

ATTR size_t
__ockl_get_local_size(uint dim)
{
    // TODO save some effort if -cl-uniform-work-group-size is used
    __constant hsa_kernel_dispatch_packet_t *p = __llvm_amdgcn_dispatch_ptr();
    uint l, g, n, d;

    switch(dim) {
    case 0:
        l = __llvm_amdgcn_workitem_id_x();
        g = __llvm_amdgcn_workgroup_id_x();
        n = p->grid_size_x;
        d = p->workgroup_size_x;
        break;
    case 1:
        l = __llvm_amdgcn_workitem_id_y();
        g = __llvm_amdgcn_workgroup_id_y();
        n = p->grid_size_y;
        d = p->workgroup_size_y;
        break;
    case 2:
        l = __llvm_amdgcn_workitem_id_z();
        g = __llvm_amdgcn_workgroup_id_z();
        n = p->grid_size_z;
        d = p->workgroup_size_z;
        break;
    default:
        l = 0;
        g = 0;
        n = 0;
        d = 1;
        break;
    }
    uint q = n / d;
    uint r = n - q*d;
    uint i = g*d + l;
    return (r > 0) & (i >= n-r) ? r : d;
}

ATTR size_t
__ockl_get_num_groups(uint dim)
{
    __constant hsa_kernel_dispatch_packet_t *p = __llvm_amdgcn_dispatch_ptr();

    uint n, d;
    switch(dim) {
    case 0:
        n = p->grid_size_x;
        d = p->workgroup_size_x;
        break;
    case 1:
        n = p->grid_size_y;
        d = p->workgroup_size_y;
        break;
    case 2:
        n = p->grid_size_z;
        d = p->workgroup_size_z;
        break;
    default:
        n = 1;
        d = 1;
        break;
    }

    uint q = n / d;

    // TODO save some effort here if -cl-uniform-work-group-size is set
    return q + (n > q*d);
}

ATTR uint
__ockl_get_work_dim(void) {
    __constant hsa_kernel_dispatch_packet_t *p = __llvm_amdgcn_dispatch_ptr();
    // XXX revist this if setup field ever changes
    return p->setup;
}

ATTR size_t
__ockl_get_enqueued_local_size(uint dim)
{
    __constant hsa_kernel_dispatch_packet_t *p = __llvm_amdgcn_dispatch_ptr();

    switch(dim) {
    case 0:
        return p->workgroup_size_x;
    case 1:
        return p->workgroup_size_y;
    case 2:
        return p->workgroup_size_z;
    default:
        return 1;
    }
}

ATTR size_t
__ockl_get_global_linear_id(void)
{
    __constant hsa_kernel_dispatch_packet_t *p = __llvm_amdgcn_dispatch_ptr();

    // XXX revisit this if setup field ever changes
    switch (p->setup) {
    case 1:
        {
            uint l0 = __llvm_amdgcn_workitem_id_x();
            uint g0 = __llvm_amdgcn_workgroup_id_x();
            uint s0 = p->workgroup_size_x;
            return g0*s0 + l0;
        }
    case 2:
        {
            uint l0 = __llvm_amdgcn_workitem_id_x();
            uint l1 = __llvm_amdgcn_workitem_id_y();
            uint g0 = __llvm_amdgcn_workgroup_id_x();
            uint g1 = __llvm_amdgcn_workgroup_id_y();
            uint s0 = p->workgroup_size_x;
            uint s1 = p->workgroup_size_y;
            uint n0 = p->grid_size_x;
            uint i0 = g0*s0 + l0;
            uint i1 = g1*s1 + l1;
            return (size_t)i1 * (size_t)n0 + i0;
        }
    case 3:
        {
            uint l0 = __llvm_amdgcn_workitem_id_x();
            uint l1 = __llvm_amdgcn_workitem_id_y();
            uint l2 = __llvm_amdgcn_workitem_id_z();
            uint g0 = __llvm_amdgcn_workgroup_id_x();
            uint g1 = __llvm_amdgcn_workgroup_id_y();
            uint g2 = __llvm_amdgcn_workgroup_id_z();
            uint s0 = p->workgroup_size_x;
            uint s1 = p->workgroup_size_y;
            uint s2 = p->workgroup_size_z;
            uint n0 = p->grid_size_x;
            uint n1 = p->grid_size_y;
            uint i0 = g0*s0 + l0;
            uint i1 = g1*s1 + l1;
            uint i2 = g2*s2 + l2;
            return ((size_t)i2 * (size_t)n1 + (size_t)i1) * (size_t)n0 + i0;
        }
    default:
        return 0;
    }
}

ATTR size_t
__ockl_get_local_linear_id(void)
{
    __constant hsa_kernel_dispatch_packet_t *p = __llvm_amdgcn_dispatch_ptr();
    return (__llvm_amdgcn_workitem_id_z()*p->workgroup_size_y +
            __llvm_amdgcn_workitem_id_y()) * p->workgroup_size_x + __llvm_amdgcn_workitem_id_x();
}

