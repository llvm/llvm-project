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

_CLC_OVERLOAD _CLC_DEF uint __clc_get_sub_group_size() {
  size_t linear_size = __clc_get_local_size(0) * __clc_get_local_size(1) *
                       __clc_get_local_size(2);
  uint remainder = linear_size % __clc_get_max_sub_group_size();
  bool full_sub_group = (remainder == 0) || (__clc_get_sub_group_id() <
                                             __clc_get_num_sub_groups() - 1);

  return full_sub_group ? __clc_get_max_sub_group_size() : remainder;
}
