//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_SUBGROUP_CLC_SUB_GROUP_SUBGROUP_H__
#define __CLC_SUBGROUP_CLC_SUB_GROUP_SUBGROUP_H__

#include "clc/internal/clc.h"

_CLC_DECL _CLC_OVERLOAD _CLC_CONST uint __clc_get_sub_group_size(void);
_CLC_DECL _CLC_OVERLOAD _CLC_CONST uint __clc_get_max_sub_group_size(void);
_CLC_DECL _CLC_OVERLOAD _CLC_CONST uint __clc_get_num_sub_groups(void);
_CLC_DECL _CLC_OVERLOAD _CLC_CONST uint __clc_get_enqueued_num_sub_groups(void);
_CLC_DECL _CLC_OVERLOAD _CLC_CONST uint __clc_get_sub_group_id(void);
_CLC_DECL _CLC_OVERLOAD _CLC_CONST uint __clc_get_sub_group_local_id(void);
_CLC_DECL _CLC_OVERLOAD _CLC_CONST int __clc_sub_group_all(int x);
_CLC_DECL _CLC_OVERLOAD _CLC_CONST int __clc_sub_group_any(int x);

#endif // __CLC_SUBGROUP_CLC_SUB_GROUP_SUBGROUP_H__
