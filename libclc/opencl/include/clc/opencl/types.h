//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_TYPES_H__
#define __CLC_OPENCL_TYPES_H__

// Copied from clang/lib/Headers/opencl-c-base.h

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

typedef enum memory_order {
  memory_order_relaxed = __ATOMIC_RELAXED,
  memory_order_acquire = __ATOMIC_ACQUIRE,
  memory_order_release = __ATOMIC_RELEASE,
  memory_order_acq_rel = __ATOMIC_ACQ_REL,
#if defined(__opencl_c_atomic_order_seq_cst)
  memory_order_seq_cst = __ATOMIC_SEQ_CST
#endif
} memory_order;

#endif // __CLC_OPENCL_TYPES_H__
