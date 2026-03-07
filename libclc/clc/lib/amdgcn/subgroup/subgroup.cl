//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/amdgpu/amdgpu_utils.h"
#include "clc/integer/clc_mul24.h"
#include "clc/shared/clc_min.h"
#include "clc/subgroup/clc_subgroup.h"
#include "clc/workitem/clc_get_enqueued_local_size.h"
#include "clc/workitem/clc_get_local_linear_id.h"
#include "clc/workitem/clc_get_local_size.h"
#include "clc/workitem/clc_get_max_sub_group_size.h"

static uint enqueued_workgroup_size() {
  return __clc_mul24((uint)__clc_get_enqueued_local_size(2),
                     __clc_mul24((uint)__clc_get_enqueued_local_size(1),
                                 (uint)__clc_get_enqueued_local_size(0)));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint __clc_get_max_sub_group_size(void) {
  return __clc_min(__builtin_amdgcn_wavefrontsize(), enqueued_workgroup_size());
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint __clc_get_num_sub_groups(void) {
  return (__clc_amdgpu_workgroup_size() + __builtin_amdgcn_wavefrontsize() -
          1) >>
         __clc_amdgpu_wavesize_log2();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint __clc_get_enqueued_num_sub_groups(void) {
  return (enqueued_workgroup_size() + __builtin_amdgcn_wavefrontsize() - 1) >>
         __clc_amdgpu_wavesize_log2();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint __clc_get_sub_group_local_id(void) {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int __clc_sub_group_all(int x) {
  return __builtin_amdgcn_ballot_w64(x) == __builtin_amdgcn_read_exec();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int __clc_sub_group_any(int x) {
  return __builtin_amdgcn_ballot_w64(x) != 0;
}
