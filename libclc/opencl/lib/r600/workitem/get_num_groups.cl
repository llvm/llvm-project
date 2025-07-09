//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/workitem/get_num_groups.h>

uint __clc_r600_get_num_groups_x(void) __asm("llvm.r600.read.ngroups.x");
uint __clc_r600_get_num_groups_y(void) __asm("llvm.r600.read.ngroups.y");
uint __clc_r600_get_num_groups_z(void) __asm("llvm.r600.read.ngroups.z");

_CLC_DEF _CLC_OVERLOAD size_t get_num_groups(uint dim) {
  switch (dim) {
  case 0:
    return __clc_r600_get_num_groups_x();
  case 1:
    return __clc_r600_get_num_groups_y();
  case 2:
    return __clc_r600_get_num_groups_z();
  default:
    return 1;
  }
}
