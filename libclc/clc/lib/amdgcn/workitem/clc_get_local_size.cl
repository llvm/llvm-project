//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/workitem/clc_get_local_size.h"
#include <amdhsa_abi.h>

_CLC_OVERLOAD _CLC_DEF size_t __clc_get_local_size(uint dim) {
  if (dim > 2)
    return 1;

  __constant amdhsa_implicit_kernarg_v5 *args =
      (__constant amdhsa_implicit_kernarg_v5 *)
          __builtin_amdgcn_implicitarg_ptr();

  uint group_ids[3] = {__builtin_amdgcn_workgroup_id_x(),
                       __builtin_amdgcn_workgroup_id_y(),
                       __builtin_amdgcn_workgroup_id_z()};

  return group_ids[dim] < args->block_count[dim] ? (size_t)args->group_size[dim]
                                                 : (size_t)args->remainder[dim];
}
