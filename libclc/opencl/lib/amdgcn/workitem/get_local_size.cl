//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/workitem/get_local_size.h>

uint __clc_amdgcn_get_local_size_x(void) __asm("llvm.r600.read.local.size.x");
uint __clc_amdgcn_get_local_size_y(void) __asm("llvm.r600.read.local.size.y");
uint __clc_amdgcn_get_local_size_z(void) __asm("llvm.r600.read.local.size.z");

_CLC_DEF _CLC_OVERLOAD size_t get_local_size(uint dim) {
  switch (dim) {
  case 0:
    return __clc_amdgcn_get_local_size_x();
  case 1:
    return __clc_amdgcn_get_local_size_y();
  case 2:
    return __clc_amdgcn_get_local_size_z();
  default:
    return 1;
  }
}
