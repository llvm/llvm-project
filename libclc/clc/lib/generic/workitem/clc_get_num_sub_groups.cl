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

_CLC_OVERLOAD _CLC_DEF uint __clc_get_num_sub_groups() {
  size_t linear_size = __clc_get_local_size(0) * __clc_get_local_size(1) *
                       __clc_get_local_size(2);
  uint sg_size = __clc_get_max_sub_group_size();
  return (uint)((linear_size + sg_size - 1) / sg_size);
}
