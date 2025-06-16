//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/workitem/clc_get_local_size.h>
#include <clc/workitem/clc_get_max_sub_group_size.h>
#include <clc/workitem/clc_get_num_sub_groups.h>
#include <clc/workitem/clc_get_sub_group_id.h>
#include <clc/workitem/clc_get_sub_group_size.h>

_CLC_OVERLOAD _CLC_DEF uint clc_get_sub_group_size() {
  if (clc_get_sub_group_id() != clc_get_num_sub_groups() - 1) {
    return clc_get_max_sub_group_size();
  }
  size_t size_x = clc_get_local_size(0);
  size_t size_y = clc_get_local_size(1);
  size_t size_z = clc_get_local_size(2);
  size_t linear_size = size_z * size_y * size_x;
  size_t uniform_groups = clc_get_num_sub_groups() - 1;
  size_t uniform_size = clc_get_max_sub_group_size() * uniform_groups;
  return linear_size - uniform_size;
}
