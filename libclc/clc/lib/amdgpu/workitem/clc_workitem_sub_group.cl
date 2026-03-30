//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/amdgpu/amdgpu_utils.h"
#include "clc/shared/clc_min.h"
#include "clc/workitem/clc_get_local_linear_id.h"
#include "clc/workitem/clc_get_max_sub_group_size.h"
#include "clc/workitem/clc_get_num_sub_groups.h"
#include "clc/workitem/clc_get_sub_group_id.h"
#include "clc/workitem/clc_get_sub_group_local_id.h"
#include "clc/workitem/clc_get_sub_group_size.h"

_CLC_OVERLOAD _CLC_DEF _CLC_CONST uint __clc_get_enqueued_num_sub_groups(void) {
  return (__clc_amdgpu_enqueued_workgroup_size() +
          __builtin_amdgcn_wavefrontsize() - 1) >>
         __clc_amdgpu_wavesize_log2();
}

_CLC_OVERLOAD _CLC_DEF uint __clc_get_max_sub_group_size(void) {
  return __clc_min(__builtin_amdgcn_wavefrontsize(),
                   __clc_amdgpu_enqueued_workgroup_size());
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST uint __clc_get_num_sub_groups(void) {
  uint group_size = __clc_amdgpu_workgroup_size();
  return (group_size + __builtin_amdgcn_wavefrontsize() - 1) >>
         __clc_amdgpu_wavesize_log2();
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST uint __clc_get_sub_group_id(void) {
  return (uint)__clc_get_local_linear_id() >> __clc_amdgpu_wavesize_log2();
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST uint __clc_get_sub_group_local_id(void) {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST uint __clc_get_sub_group_size(void) {
  uint wavesize = __builtin_amdgcn_wavefrontsize();
  uint lid = (uint)__clc_get_local_linear_id();
  return __clc_min(wavesize,
                   __clc_amdgpu_workgroup_size() - (lid & ~(wavesize - 1)));
}
