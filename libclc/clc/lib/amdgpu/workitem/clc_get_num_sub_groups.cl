//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/amdgpu/amdgpu_utils.h"
#include "clc/subgroup/clc_subgroup.h"

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint __clc_get_num_sub_groups(void) {
  uint group_size = __clc_amdgpu_workgroup_size();
  return (group_size + __builtin_amdgcn_wavefrontsize() - 1) >>
         __clc_amdgpu_wavesize_log2();
}
