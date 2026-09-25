//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/amdgpu/amdgpu_utils.h"
#include "clc/workitem/clc_get_local_linear_id.h"
#include "clc/workitem/clc_get_sub_group_id.h"

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint __clc_get_sub_group_id(void) {
  return (uint)__clc_get_local_linear_id() >> __clc_amdgpu_wavesize_log2();
}
