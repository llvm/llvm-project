#include "ockl.h"
#include "irif.h"

#define ATTR __attribute__((always_inline, const))
#define ATTR2 __attribute__((always_inline))

ATTR long
amp_get_global_id(int dim)
{
  return __ockl_get_global_id(dim);
}

ATTR long
amp_get_global_size(int dim)
{
  return __ockl_get_global_size(dim);
}

ATTR long
amp_get_local_id(int dim)
{
  return __ockl_get_local_id(dim);
}

ATTR long
amp_get_num_groups(int dim)
{
  return __ockl_get_num_groups(dim);
}

ATTR long
amp_get_group_id(int dim)
{
  return __ockl_get_group_id(dim);
}

ATTR long
amp_get_local_size(int dim)
{
  return __ockl_get_local_size(dim);
}

ATTR long
hc_get_grid_size(int dim)
{
  return __ockl_get_global_size(dim);
}

ATTR long
hc_get_workitem_absolute_id(int dim)
{
  return __ockl_get_global_id(dim);
}

ATTR long
hc_get_workitem_id(int dim)
{
  return __ockl_get_local_id(dim);
}

ATTR long
hc_get_num_groups(int dim)
{
  return __ockl_get_num_groups(dim);
}

ATTR long
hc_get_group_id(int dim)
{
  return __ockl_get_group_id(dim);
}

ATTR long
hc_get_group_size(int dim)
{
  return __ockl_get_local_size(dim);
}

ATTR2 void
hc_barrier(int n)
{
  __llvm_amdgcn_s_waitcnt(0);
  __llvm_amdgcn_s_dcache_wb();
  __llvm_amdgcn_s_barrier();
}

ATTR2 void
amp_barrier(int n)
{
  hc_barrier(n);
}
