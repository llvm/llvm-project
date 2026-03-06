//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/opencl-base.h>

_CLC_DEF _CLC_OVERLOAD void sub_group_barrier(cl_mem_fence_flags flags,
                                              memory_scope scope) {
  __builtin_amdgcn_wave_barrier();

  if (flags)
    atomic_work_item_fence(flags, memory_order_acq_rel, scope);
}

_CLC_DEF _CLC_OVERLOAD void sub_group_barrier(cl_mem_fence_flags flags) {
  sub_group_barrier(flags, memory_scope_sub_group);
}
