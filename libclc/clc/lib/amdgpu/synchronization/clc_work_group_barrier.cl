//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/mem_fence/clc_mem_fence.h>
#include <clc/synchronization/clc_work_group_barrier.h>

_CLC_OVERLOAD _CLC_DEF void
__clc_work_group_barrier(int memory_scope,
                         __CLC_MemorySemantics memory_semantics) {
  if (memory_semantics == 0) {
    __builtin_amdgcn_s_barrier();
  } else {
    uint seq_cst_mask = __CLC_MEMORY_GLOBAL | __CLC_MEMORY_LOCAL;
    int memory_order_before = (memory_semantics & seq_cst_mask) == seq_cst_mask
                                  ? __ATOMIC_SEQ_CST
                                  : __ATOMIC_RELEASE;
    int memory_order_after = (memory_semantics & seq_cst_mask) == seq_cst_mask
                                 ? __ATOMIC_SEQ_CST
                                 : __ATOMIC_ACQUIRE;

    __clc_mem_fence(memory_scope, memory_order_before, memory_semantics);
    __builtin_amdgcn_s_barrier();
    __clc_mem_fence(memory_scope, memory_order_after, memory_semantics);
  }
}
