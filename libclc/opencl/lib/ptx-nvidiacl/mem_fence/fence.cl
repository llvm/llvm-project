//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/mem_fence/clc_mem_fence.h>
#include <clc/opencl/explicit_fence/explicit_memory_fence.h>
#include <clc/opencl/synchronization/utils.h>

_CLC_DEF _CLC_OVERLOAD void mem_fence(cl_mem_fence_flags flags) {
  int memory_scope = getCLCMemoryScope(flags);
  int memory_order = __ATOMIC_SEQ_CST;
  __clc_mem_fence(memory_scope, memory_order);
}

// We do not have separate mechanism for read and write fences.
_CLC_DEF _CLC_OVERLOAD void read_mem_fence(cl_mem_fence_flags flags) {
  mem_fence(flags);
}

_CLC_DEF _CLC_OVERLOAD void write_mem_fence(cl_mem_fence_flags flags) {
  mem_fence(flags);
}
