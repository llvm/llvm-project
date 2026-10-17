//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_SUBGROUP_CLC_SUB_GROUP_NON_UNIFORM_REDUCE_H__
#define __CLC_SUBGROUP_CLC_SUB_GROUP_NON_UNIFORM_REDUCE_H__

#include "clc/internal/clc.h"

#define __CLC_BODY "clc/subgroup/clc_sub_group_non_uniform_reduce_decl.inc"
#include "clc/integer/gentype.inc"

#define __CLC_BODY "clc/subgroup/clc_sub_group_non_uniform_reduce_decl.inc"
#include "clc/math/gentype.inc"

_CLC_DECL _CLC_OVERLOAD int
__clc_sub_group_non_uniform_reduce_logical_and(int x);

_CLC_DECL _CLC_OVERLOAD int
__clc_sub_group_non_uniform_reduce_logical_or(int x);

_CLC_DECL _CLC_OVERLOAD int
__clc_sub_group_non_uniform_reduce_logical_xor(int x);

#endif // __CLC_SUBGROUP_CLC_SUB_GROUP_NON_UNIFORM_REDUCE_H__
