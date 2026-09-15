//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/workitem/clc_get_enqueued_local_size.h"
#include "clc/workitem/clc_get_global_linear_id.h"
#include "clc/workitem/clc_get_global_size.h"
#include "clc/workitem/clc_get_group_id.h"
#include "clc/workitem/clc_get_local_id.h"
#include "clc/workitem/clc_get_work_dim.h"

static size_t get_global_id_no_offset(uint dim) {
  return __clc_get_group_id(dim) * __clc_get_enqueued_local_size(dim) +
         __clc_get_local_id(dim);
}

static size_t get_global_linear_id_1d() { return get_global_id_no_offset(0); }

static size_t get_global_linear_id_2d() {
  return get_global_id_no_offset(1) * __clc_get_global_size(0) +
         get_global_linear_id_1d();
}

static size_t get_global_linear_id_3d() {
  return (get_global_id_no_offset(2) * __clc_get_global_size(1) +
          get_global_id_no_offset(1)) *
             __clc_get_global_size(0) +
         get_global_id_no_offset(0);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST size_t __clc_get_global_linear_id() {
  switch (__clc_get_work_dim()) {
  case 1:
    return get_global_linear_id_1d();
  case 2:
    return get_global_linear_id_2d();
  case 3:
    return get_global_linear_id_3d();
  default:
    return 0;
  }
}
