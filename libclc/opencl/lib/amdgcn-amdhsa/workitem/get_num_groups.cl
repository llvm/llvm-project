//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/workitem/get_global_size.h>
#include <clc/opencl/workitem/get_local_size.h>
#include <clc/opencl/workitem/get_num_groups.h>

_CLC_DEF _CLC_OVERLOAD size_t get_num_groups(uint dim) {
  size_t global_size = get_global_size(dim);
  size_t local_size = get_local_size(dim);
  size_t num_groups = global_size / local_size;
  if (global_size % local_size != 0) {
    num_groups++;
  }
  return num_groups;
}
