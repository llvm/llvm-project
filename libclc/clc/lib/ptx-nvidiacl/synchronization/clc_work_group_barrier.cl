//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/synchronization/clc_work_group_barrier.h>

_CLC_OVERLOAD _CLC_DEF void
__clc_work_group_barrier(int memory_scope, int memory_order,
                         __CLC_MemorySemantics memory_semantics) {
  __syncthreads();
}
