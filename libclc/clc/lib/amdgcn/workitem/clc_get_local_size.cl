//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/workitem/clc_get_local_size.h>

_CLC_OVERLOAD _CLC_DEF size_t __clc_get_local_size(uint dim) {
  switch (dim) {
  case 0:
    return __builtin_amdgcn_workgroup_size_x();
  case 1:
    return __builtin_amdgcn_workgroup_size_y();
  case 2:
    return __builtin_amdgcn_workgroup_size_z();
  default:
    return 1;
  }
}
