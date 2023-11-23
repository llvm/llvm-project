// REQUIRES: amdgpu-registered-target
//
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -O3 -S -o - %s \
// RUN:   -emit-llvm \
// RUN:   | FileCheck -check-prefixes=COMMON,UNSAFE-INT-DEFAULT %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -O3 -S -o - %s \
// RUN:   -emit-llvm -mamdgpu-fine-grained-mem \
// RUN:   | FileCheck -check-prefixes=COMMON,UNSAFE-INT-ON %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -O3 -S -o - %s \
// RUN:   -emit-llvm -mno-amdgpu-fine-grained-mem \
// RUN:   | FileCheck -check-prefixes=COMMON,UNSAFE-INT-OFF %s

// Check AMDGCN ISA generation.

// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -O3 -S -o - %s \
// RUN:   | FileCheck -check-prefixes=COMMON-ISA,ISA-DEFAULT %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -O3 -S -o - %s \
// RUN:   -mamdgpu-fine-grained-mem \
// RUN:   | FileCheck -check-prefixes=COMMON-ISA,ISA-ON %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -O3 -S -o - %s \
// RUN:   -mno-amdgpu-fine-grained-mem \
// RUN:   | FileCheck -check-prefixes=COMMON-ISA,ISA-OFF %s

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

typedef enum memory_order {
  memory_order_relaxed = __ATOMIC_RELAXED,
  memory_order_acquire = __ATOMIC_ACQUIRE,
  memory_order_release = __ATOMIC_RELEASE,
  memory_order_acq_rel = __ATOMIC_ACQ_REL,
  memory_order_seq_cst = __ATOMIC_SEQ_CST
} memory_order;

typedef enum memory_scope {
  memory_scope_work_item = __OPENCL_MEMORY_SCOPE_WORK_ITEM,
  memory_scope_work_group = __OPENCL_MEMORY_SCOPE_WORK_GROUP,
  memory_scope_device = __OPENCL_MEMORY_SCOPE_DEVICE,
  memory_scope_all_svm_devices = __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
#if defined(cl_intel_subgroups) || defined(cl_khr_subgroups)
  memory_scope_sub_group = __OPENCL_MEMORY_SCOPE_SUB_GROUP
#endif
} memory_scope;

// COMMON-ISA: kern:
// ISA-ON: flat_atomic_cmpswap v{{[0-9]+}},  v[{{[0-9]+}}:{{[0-9]+}}],  v[{{[0-9]+}}:{{[0-9]+}}] glc
// ISA-OFF: flat_atomic_xor v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc
// ISA-DEFAULT: flat_atomic_xor v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc
kernel void kern(global atomic_int *x, int y, global int *z) {
  *z = __opencl_atomic_fetch_xor(x, y, memory_order_seq_cst, memory_scope_work_group);
}

// COMMON: define{{.*}} amdgpu_kernel void @kern
// COMMON: atomicrmw xor ptr addrspace(1) %x, i32 %y syncscope("workgroup") seq_cst, align 4

// UNSAFE-INT-ON-SAME: !amdgpu.atomic ![[REF:[0-9]+]]
// UNSAFE-INT-ON: ![[REF]] = !{![[REF2:[0-9]+]]}
// UNSAFE-INT-ON: ![[REF2]] = !{!"fine_grained", i32 1}

// UNSAFE-INT-OFF-NOT: !amdgpu.atomic

// UNSAFE-INT-DEFAULT-NOT: !amdgpu.atomic
