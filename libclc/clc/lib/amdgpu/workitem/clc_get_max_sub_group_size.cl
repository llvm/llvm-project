//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/amdgpu/amdgpu_utils.h"
#include "clc/shared/clc_min.h"
#include "clc/workitem/clc_get_max_sub_group_size.h"

_CLC_OVERLOAD _CLC_DEF uint __clc_get_max_sub_group_size(void) {
  return __clc_min(__builtin_amdgcn_wavefrontsize(),
                   __clc_amdgpu_enqueued_workgroup_size());
}
