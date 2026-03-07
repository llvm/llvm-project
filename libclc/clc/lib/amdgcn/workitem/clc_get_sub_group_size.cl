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
#include "clc/workitem/clc_get_local_linear_id.h"
#include "clc/workitem/clc_get_local_size.h"

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint __clc_get_sub_group_size(void) {
  uint wavesize = __builtin_amdgcn_wavefrontsize();
  uint lid = (uint)__clc_get_local_linear_id();
  return __clc_min(wavesize,
                   __clc_amdgpu_workgroup_size() - (lid & ~(wavesize - 1)));
}
