//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/amdgpu/amdgpu_utils.h"
#include "clc/workitem/clc_get_sub_group_local_id.h"

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint __clc_get_enqueued_num_sub_groups(void) {
  return (__clc_amdgpu_enqueued_workgroup_size() +
          __builtin_amdgcn_wavefrontsize() - 1) >>
         __clc_amdgpu_wavesize_log2();
}
