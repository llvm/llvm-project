//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_SYNCHRONIZATION_UTILS_H__
#define __CLC_OPENCL_SYNCHRONIZATION_UTILS_H__

#include <clc/internal/clc.h>
#include <clc/mem_fence/clc_mem_semantic.h>
#include <clc/opencl/synchronization/cl_mem_fence_flags.h>

_CLC_INLINE int getCLCMemoryScope(cl_mem_fence_flags flag) {
  int memory_scope = 0;
  if (flag & CLK_GLOBAL_MEM_FENCE)
    memory_scope |= __MEMORY_SCOPE_DEVICE;
  if (flag & CLK_LOCAL_MEM_FENCE)
    memory_scope |= __MEMORY_SCOPE_WRKGRP;
  return memory_scope;
}

_CLC_INLINE MemorySemantics getCLCMemorySemantics(cl_mem_fence_flags flag) {
  if ((flag & CLK_LOCAL_MEM_FENCE) && (flag & CLK_GLOBAL_MEM_FENCE))
    return MEMORY_LOCAL | MEMORY_GLOBAL;
  if (flag & CLK_LOCAL_MEM_FENCE)
    return MEMORY_LOCAL;
  if (flag & CLK_GLOBAL_MEM_FENCE)
    return MEMORY_GLOBAL;
  __builtin_unreachable();
}

#endif // __CLC_OPENCL_SYNCHRONIZATION_UTILS_H__
