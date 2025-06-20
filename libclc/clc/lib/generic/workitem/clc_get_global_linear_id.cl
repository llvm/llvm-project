//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/workitem/clc_get_global_id.h>
#include <clc/workitem/clc_get_global_linear_id.h>
#include <clc/workitem/clc_get_global_offset.h>
#include <clc/workitem/clc_get_global_size.h>
#include <clc/workitem/clc_get_work_dim.h>

_CLC_OVERLOAD _CLC_DEF size_t __clc_get_global_linear_id() {
  uint dim = __clc_get_work_dim();
  switch (dim) {
  default:
  case 1:
    return __clc_get_global_id(0) - __clc_get_global_offset(0);
  case 2:
    return (__clc_get_global_id(1) - __clc_get_global_offset(1)) *
               __clc_get_global_size(0) +
           (__clc_get_global_id(0) - __clc_get_global_offset(0));
  case 3:
    return ((__clc_get_global_id(2) - __clc_get_global_offset(2)) *
            __clc_get_global_size(1) * __clc_get_global_size(0)) +
           ((__clc_get_global_id(1) - __clc_get_global_offset(1)) *
            __clc_get_global_size(0)) +
           (__clc_get_global_id(0) - __clc_get_global_offset(0));
  }
}
