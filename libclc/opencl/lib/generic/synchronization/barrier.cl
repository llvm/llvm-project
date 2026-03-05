//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/synchronization/utils.h>
#include <clc/synchronization/clc_work_group_barrier.h>

_CLC_DEF _CLC_OVERLOAD void work_group_barrier(cl_mem_fence_flags flags,
                                               memory_scope scope) {
  int memory_order = __ATOMIC_SEQ_CST;
  __CLC_MemorySemantics memory_semantics = __opencl_get_memory_semantics(flags);
  __clc_work_group_barrier(scope, memory_order, memory_semantics);
}

_CLC_DEF _CLC_OVERLOAD void work_group_barrier(cl_mem_fence_flags flags) {
  work_group_barrier(flags, memory_scope_work_group);
}

_CLC_DEF _CLC_OVERLOAD void barrier(cl_mem_fence_flags flags) {
  work_group_barrier(flags);
}
