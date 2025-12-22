//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_MEM_FENCE_CLC_MEM_FENCE_H__
#define __CLC_MEM_FENCE_CLC_MEM_FENCE_H__

#include <clc/internal/clc.h>
#include <clc/mem_fence/clc_mem_semantic.h>

_CLC_OVERLOAD _CLC_DECL void
__clc_mem_fence(int memory_scope, int memory_order,
                __CLC_MemorySemantics memory_semantics);

#endif // __CLC_MEM_FENCE_CLC_MEM_FENCE_H__
