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
#include <clc/mem_fence/clc_mem_scope_semantics.h>
#include <clc/opencl/synchronization/cl_mem_fence_flags.h>

_CLC_INLINE Scope getCLCScope(memory_scope memory_scope) {
  switch (memory_scope) {
  case memory_scope_work_item:
    return Invocation;
#if defined(cl_intel_subgroups) || defined(cl_khr_subgroups) ||                \
    defined(__opencl_c_subgroups)
  case memory_scope_sub_group:
    return Subgroup;
#endif
  case memory_scope_work_group:
    return Workgroup;
  case memory_scope_device:
    return Device;
  default:
    break;
  }
#ifdef __opencl_c_atomic_scope_all_devices
  return CrossDevice;
#else
  return Device;
#endif
}

_CLC_INLINE MemorySemantics getCLCMemorySemantics(cl_mem_fence_flags flag) {
  MemorySemantics semantics = SequentiallyConsistent;
  if (flag & CLK_GLOBAL_MEM_FENCE)
    semantics |= CrossWorkgroupMemory;
  if (flag & CLK_LOCAL_MEM_FENCE)
    semantics |= WorkgroupMemory;
  if (flag & CLK_IMAGE_MEM_FENCE)
    semantics |= ImageMemory;
  return semantics;
}

#endif // __CLC_OPENCL_SYNCHRONIZATION_UTILS_H__
