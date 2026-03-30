//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/amdgpu/amdgpu_utils.h"
#include "clc/subgroup/clc_subgroup.h"

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int __clc_sub_group_all(int x) {
  return __builtin_amdgcn_ballot_w64(x) == __builtin_amdgcn_read_exec();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int __clc_sub_group_any(int x) {
  return __builtin_amdgcn_ballot_w64(x) != 0;
}
