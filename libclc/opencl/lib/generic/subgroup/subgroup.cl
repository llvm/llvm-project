//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/subgroup/clc_subgroup.h"

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint get_sub_group_size(void) {
  return __clc_get_sub_group_size();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint get_max_sub_group_size(void) {
  return __clc_get_max_sub_group_size();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint get_num_sub_groups(void) {
  return __clc_get_num_sub_groups();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint get_enqueued_num_sub_groups(void) {
  return __clc_get_enqueued_num_sub_groups();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint get_sub_group_id(void) {
  return __clc_get_sub_group_id();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint get_sub_group_local_id(void) {
  return __clc_get_sub_group_local_id();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int sub_group_all(int x) {
  return __clc_sub_group_all(x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int sub_group_any(int x) {
  return __clc_sub_group_any(x);
}
