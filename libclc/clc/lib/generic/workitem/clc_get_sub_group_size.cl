//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/shared/clc_min.h"
#include "clc/workitem/clc_get_local_linear_id.h"
#include "clc/workitem/clc_get_local_size.h"
#include "clc/workitem/clc_get_max_sub_group_size.h"
#include "clc/workitem/clc_get_sub_group_size.h"

_CLC_OVERLOAD _CLC_DEF uint __clc_get_sub_group_size() {
  uint local_linear_size = (uint)__clc_get_local_size(0) *
                           (uint)__clc_get_local_size(1) *
                           (uint)__clc_get_local_size(2);
  uint max_sg_size = __clc_get_max_sub_group_size();
  // Assume max_sg_size is power of 2.
  uint remainder = local_linear_size & (max_sg_size - 1);
  if (remainder == 0)
    return max_sg_size;
  uint lid = (uint)__clc_get_local_linear_id();
  return __clc_min(max_sg_size, local_linear_size - (lid & ~(max_sg_size - 1)));
}
