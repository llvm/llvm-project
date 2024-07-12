// RUN: %clang_cc1 %s -cl-std=CL2.0 -O0 -triple=amdgcn-amd-amdhsa -target-cpu gfx90a \
// RUN:     -Rpass=atomic-expand -S -o - 2>&1 | \
// RUN:     FileCheck %s --check-prefix=REMARK

// RUN: %clang_cc1 %s -cl-std=CL2.0 -O0 -triple=amdgcn-amd-amdhsa -target-cpu gfx90a \
// RUN:     -Rpass=atomic-expand -emit-llvm -o - 2>&1 | \
// RUN:     FileCheck %s --check-prefix=GFX90A-CAS

// REQUIRES: amdgpu-registered-target

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

// REMARK: remark: A compare and swap loop was generated for an atomic fadd operation at workgroup-one-as memory scope [-Rpass=atomic-expand]
// REMARK: remark: A compare and swap loop was generated for an atomic fadd operation at agent-one-as memory scope [-Rpass=atomic-expand]
// REMARK: remark: A compare and swap loop was generated for an atomic fadd operation at one-as memory scope [-Rpass=atomic-expand]
// REMARK: remark: A compare and swap loop was generated for an atomic fadd operation at wavefront-one-as memory scope [-Rpass=atomic-expand]
// GFX90A-CAS-LABEL: @atomic_cas
// GFX90A-CAS: atomicrmw fadd ptr addrspace(1) {{.*}} syncscope("workgroup-one-as") monotonic
// GFX90A-CAS: atomicrmw fadd ptr addrspace(1) {{.*}} syncscope("agent-one-as") monotonic
// GFX90A-CAS: atomicrmw fadd ptr addrspace(1) {{.*}} syncscope("one-as") monotonic
// GFX90A-CAS: atomicrmw fadd ptr addrspace(1) {{.*}} syncscope("wavefront-one-as") monotonic
float atomic_cas(__global atomic_float *d, float a) {
  float ret1 = __opencl_atomic_fetch_add(d, a, memory_order_relaxed, memory_scope_work_group);
  float ret2 = __opencl_atomic_fetch_add(d, a, memory_order_relaxed, memory_scope_device);
  float ret3 = __opencl_atomic_fetch_add(d, a, memory_order_relaxed, memory_scope_all_svm_devices);
  float ret4 = __opencl_atomic_fetch_add(d, a, memory_order_relaxed, memory_scope_sub_group);
}
