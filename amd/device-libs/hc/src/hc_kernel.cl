#include "ockl.h"

long amp_get_global_id(int dim)
{
  return __ockl_get_global_id(dim);
}

long amp_get_global_size(int dim)
{
  return __ockl_get_global_size(dim);
}

long amp_get_local_id(int dim)
{
  return __ockl_get_local_id(dim);
}

long amp_get_num_groups(int dim)
{
  return __ockl_get_num_groups(dim);
}

long amp_get_group_id(int dim)
{
  return __ockl_get_group_id(dim);
}

long amp_get_local_size(int dim)
{
  return __ockl_get_local_size(dim);
}

long hc_get_grid_size(int dim)
{
  return __ockl_get_global_size(dim);
}

long hc_get_workitem_absolute_id(int dim)
{
  return __ockl_get_global_id(dim);
}

long hc_get_workitem_id(int dim)
{
  return __ockl_get_local_id(dim);
}

long hc_get_num_groups(int dim)
{
  return __ockl_get_num_groups(dim);
}

long hc_get_group_id(int dim)
{
  return __ockl_get_group_id(dim);
}

long hc_get_group_size(int dim)
{
  return __ockl_get_local_size(dim);
}

void amp_barrier(int n)
{
  __ockl_barrier(n);
}

void hc_barrier(int n)
{
  __ockl_barrier(n);
}
