//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/opencl/opencl-base.h"

_CLC_DEF _CLC_OVERLOAD void wait_group_events(int n, __private event_t *evs) {
  (void)n;
  (void)evs;
  work_group_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE,
                     memory_scope_work_group);
}

#if _CLC_GENERIC_AS_SUPPORTED

_CLC_DEF _CLC_OVERLOAD void wait_group_events(int n, event_t *evs) {
  (void)n;
  (void)evs;
  work_group_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE,
                     memory_scope_work_group);
}

#endif
