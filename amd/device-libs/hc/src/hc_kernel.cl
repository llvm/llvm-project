#include "ockl.h"
#include "irif.h"
#include "device_amd_hsa.h"

#define ATTR __attribute__((always_inline, const))
#define ATTR2 __attribute__((always_inline))

ATTR uint
amp_get_global_id(int dim)
{
  __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();
  uint l, g, s;

  switch(dim) {
  case 0:
    l = __builtin_amdgcn_workitem_id_x();
    g = __builtin_amdgcn_workgroup_id_x();
    s = p->workgroup_size_x;
    break;
  case 1:
    l = __builtin_amdgcn_workitem_id_y();
    g = __builtin_amdgcn_workgroup_id_y();
    s = p->workgroup_size_y;
    break;
  case 2:
    l = __builtin_amdgcn_workitem_id_z();
    g = __builtin_amdgcn_workgroup_id_z();
    s = p->workgroup_size_z;
    break;
  default:
    l = 0;
    g = 0;
    s = 1;
    break;
  }

  return (g*s + l);
}

ATTR uint
amp_get_global_size(int dim)
{
    __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();

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

ATTR uint
amp_get_local_id(int dim)
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

ATTR uint
amp_get_num_groups(int dim)
{
    __constant hsa_kernel_dispatch_packet_t *p = __builtin_amdgcn_dispatch_ptr();

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

    return n / d;
}

ATTR uint
amp_get_group_id(int dim)
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

ATTR uint
amp_get_local_size(int dim)
{
    return __ockl_get_local_size(dim);
}

ATTR uint
hc_get_grid_size(int dim)
{
    return amp_get_global_size(dim);
}

ATTR uint
hc_get_workitem_absolute_id(int dim)
{
    return amp_get_global_id(dim);
}

ATTR uint
hc_get_workitem_id(int dim)
{
    return amp_get_local_id(dim);
}

ATTR uint
hc_get_num_groups(int dim)
{
    return amp_get_num_groups(dim);
}

ATTR uint
hc_get_group_id(int dim)
{
    return amp_get_group_id(dim);
}

ATTR uint
hc_get_group_size(int dim)
{
    return amp_get_local_size(dim);
}

ATTR2 void
hc_work_group_barrier(cl_mem_fence_flags flags, memory_scope scope)
{
    if (flags) {
        atomic_work_item_fence(flags, memory_order_release, scope);
        __builtin_amdgcn_s_barrier();
        atomic_work_item_fence(flags, memory_order_acquire, scope);
    } else {
        __builtin_amdgcn_s_barrier();
    }
}

ATTR2 void
hc_barrier(int n)
{
  hc_work_group_barrier((cl_mem_fence_flags)n, memory_scope_work_group);
}

ATTR2 void
amp_barrier(int n)
{
  hc_barrier(n);
}

