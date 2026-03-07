//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/opencl/synchronization/utils.h"
#include "clc/opencl/utils.h"
#include "clc/synchronization/clc_sub_group_barrier.h"

_CLC_DEF _CLC_OVERLOAD void sub_group_barrier(cl_mem_fence_flags flags,
                                              memory_scope scope) {
  __CLC_MemorySemantics memory_semantics = __opencl_get_memory_semantics(flags);
  __clc_sub_group_barrier(memory_semantics,
                          __opencl_get_clang_memory_scope(scope));
}

_CLC_DEF _CLC_OVERLOAD void sub_group_barrier(cl_mem_fence_flags flags) {
  __CLC_MemorySemantics memory_semantics = __opencl_get_memory_semantics(flags);
  __clc_sub_group_barrier(memory_semantics);
}
