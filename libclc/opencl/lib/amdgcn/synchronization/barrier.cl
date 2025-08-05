//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/synchronization/barrier.h>
#include <clc/opencl/synchronization/utils.h>
#include <clc/synchronization/clc_work_group_barrier.h>

_CLC_DEF _CLC_OVERLOAD void barrier(cl_mem_fence_flags flags) {
  int memory_scope = getCLCMemoryScope(flags);
  int memory_order = __ATOMIC_SEQ_CST;
  __clc_work_group_barrier(memory_scope, memory_order);
}
