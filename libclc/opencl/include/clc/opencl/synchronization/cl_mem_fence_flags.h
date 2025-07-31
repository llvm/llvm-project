//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_SYNCHRONIZATION_CL_MEM_FENCE_FLAGS_H__
#define __CLC_OPENCL_SYNCHRONIZATION_CL_MEM_FENCE_FLAGS_H__

typedef uint cl_mem_fence_flags;

// Copied from
// https://github.com/llvm/llvm-project/blob/08e40c12fa0c/clang/lib/Headers/opencl-c-base.h#L390
typedef enum memory_scope {
  memory_scope_work_item = __OPENCL_MEMORY_SCOPE_WORK_ITEM,
  memory_scope_work_group = __OPENCL_MEMORY_SCOPE_WORK_GROUP,
  memory_scope_device = __OPENCL_MEMORY_SCOPE_DEVICE,
#if defined(__opencl_c_atomic_scope_all_devices)
  memory_scope_all_svm_devices = __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
#if (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 || __OPENCL_CPP_VERSION__ >= 202100)
  memory_scope_all_devices = memory_scope_all_svm_devices,
#endif // (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 || __OPENCL_CPP_VERSION__ >=
       // 202100)
#endif // defined(__opencl_c_atomic_scope_all_devices)
/**
 * Subgroups have different requirements on forward progress, so just test
 * all the relevant macros.
 * CL 3.0 sub-groups "they are not guaranteed to make independent forward
 * progress" KHR subgroups "Subgroups within a workgroup are independent, make
 * forward progress with respect to each other"
 */
#if defined(cl_intel_subgroups) || defined(cl_khr_subgroups) ||                \
    defined(__opencl_c_subgroups)
  memory_scope_sub_group = __OPENCL_MEMORY_SCOPE_SUB_GROUP
#endif
} memory_scope;

#define CLK_LOCAL_MEM_FENCE 1
#define CLK_GLOBAL_MEM_FENCE 2
#define CLK_IMAGE_MEM_FENCE 4

#endif // __CLC_OPENCL_SYNCHRONIZATION_CL_MEM_FENCE_FLAGS_H__
