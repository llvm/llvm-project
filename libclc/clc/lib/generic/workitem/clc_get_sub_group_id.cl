//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/workitem/clc_get_local_id.h>
#include <clc/workitem/clc_get_local_size.h>
#include <clc/workitem/clc_get_max_sub_group_size.h>
#include <clc/workitem/clc_get_sub_group_id.h>

_CLC_OVERLOAD _CLC_DEF uint __clc_get_sub_group_id() {
  // sreg.warpid is volatile and doesn't represent virtual warp index
  // see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
  size_t id_x = __clc_get_local_id(0);
  size_t id_y = __clc_get_local_id(1);
  size_t id_z = __clc_get_local_id(2);
  size_t size_x = __clc_get_local_size(0);
  size_t size_y = __clc_get_local_size(1);
  size_t size_z = __clc_get_local_size(2);
  uint sg_size = __clc_get_max_sub_group_size();
  return (id_z * size_y * size_x + id_y * size_x + id_x) / sg_size;
}
