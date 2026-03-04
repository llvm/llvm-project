/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "ockl.h"

__attribute__((target("cumode,gfx1250-insts"), const)) uint
__ockl_cluster_num_workgroups(int dim)
{
    switch (dim) {
    case 0:
        return __builtin_amdgcn_cluster_workgroup_max_id_x() + 1;
    case 1:
        return __builtin_amdgcn_cluster_workgroup_max_id_y() + 1;
    case 2:
        return __builtin_amdgcn_cluster_workgroup_max_id_z() + 1;
    default:
        return 1;
    }
}

__attribute__((target("cumode,gfx1250-insts"), const)) uint
__ockl_cluster_workgroup_id(int dim)
{
    switch (dim) {
    case 0:
        return __builtin_amdgcn_cluster_workgroup_id_x();
    case 1:
        return __builtin_amdgcn_cluster_workgroup_id_y();
    case 2:
        return __builtin_amdgcn_cluster_workgroup_id_z();
    default:
        return 0;
    }
}

__attribute__((target("cumode,gfx1250-insts"), const)) uint
__ockl_cluster_flat_num_workgroups(void)
{
    return __builtin_amdgcn_cluster_workgroup_max_flat_id() + 1;
}
