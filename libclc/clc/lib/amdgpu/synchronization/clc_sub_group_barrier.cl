//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/mem_fence/clc_mem_fence.h"
#include "clc/synchronization/clc_sub_group_barrier.h"

_CLC_DEF _CLC_OVERLOAD void
__clc_sub_group_barrier(__CLC_MemorySemantics memory_semantics, int scope) {
  __builtin_amdgcn_wave_barrier();

  if (memory_semantics)
    __clc_mem_fence(scope, __ATOMIC_ACQ_REL, memory_semantics);
}
