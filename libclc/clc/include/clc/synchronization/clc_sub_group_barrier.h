//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_SUBGROUP_CLC_SUB_GROUP_BARRIER_H__
#define __CLC_SUBGROUP_CLC_SUB_GROUP_BARRIER_H__

#include "clc/internal/clc.h"
#include "clc/mem_fence/clc_mem_semantic.h"

_CLC_DECL _CLC_OVERLOAD void
__clc_sub_group_barrier(__CLC_MemorySemantics memory_semantics,
                        int memory_scope);
_CLC_DECL _CLC_OVERLOAD void
__clc_sub_group_barrier(__CLC_MemorySemantics memory_semantics);

#endif // __CLC_SUBGROUP_CLC_SUB_GROUP_BARRIER_H__
