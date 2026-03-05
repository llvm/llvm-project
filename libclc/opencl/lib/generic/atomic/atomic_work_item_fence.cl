//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/mem_fence/clc_mem_fence.h"
#include "clc/opencl/synchronization/utils.h"
#include "clc/opencl/utils.h"

_CLC_OVERLOAD _CLC_DEF void atomic_work_item_fence(cl_mem_fence_flags flags,
                                                   memory_order order,
                                                   memory_scope scope) {
  __clc_mem_fence(__opencl_get_clang_memory_scope(scope), order,
                  __opencl_get_memory_semantics(flags));
}
