// RUN: %clang_cc1 -cl-std=CL2.0 -O0 -triple=amdgcn-amd-amdhsa -target-cpu gfx90a \
// RUN:     -Rpass=si-lower -munsafe-fp-atomics %s -emit-llvm -o - 2>&1 | \
// RUN:     FileCheck %s --check-prefix=GFX90A-HW

// RUN: %clang_cc1 -cl-std=CL2.0 -O0 -triple=amdgcn-amd-amdhsa -target-cpu gfx90a \
// RUN:     -Rpass=si-lower -munsafe-fp-atomics %s -S -o - 2>&1 | \
// RUN:     FileCheck %s --check-prefix=GFX90A-HW-REMARK


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

// GFX90A-HW-REMARK: Hardware instruction generated for atomic fadd operation at memory scope wavefront-one-as due to an unsafe request. [-Rpass=si-lower]
// GFX90A-HW-REMARK: Hardware instruction generated for atomic fadd operation at memory scope agent-one-as due to an unsafe request. [-Rpass=si-lower]
// GFX90A-HW-REMARK: Hardware instruction generated for atomic fadd operation at memory scope workgroup-one-as due to an unsafe request. [-Rpass=si-lower]

// GFX90A-HW-REMARK: global_atomic_add_f32 v0, v[0:1], v2, off glc
// GFX90A-HW-REMARK: global_atomic_add_f32 v0, v[0:1], v2, off glc
// GFX90A-HW-REMARK: global_atomic_add_f32 v0, v[0:1], v2, off glc
// GFX90A-HW-LABEL: @atomic_unsafe_hw
// GFX90A-HW:   atomicrmw fadd ptr addrspace(1) %{{.*}}, float %{{.*}} syncscope("workgroup-one-as") monotonic, align 4
// GFX90A-HW:   atomicrmw fadd ptr addrspace(1) %{{.*}}, float %{{.*}} syncscope("agent-one-as") monotonic, align 4
// GFX90A-HW:   atomicrmw fadd ptr addrspace(1) %{{.*}}, float %{{.*}} syncscope("wavefront-one-as") monotonic, align 4
void atomic_unsafe_hw(__global atomic_float *d, float a) {
  float ret1 = __opencl_atomic_fetch_add(d, a, memory_order_relaxed, memory_scope_work_group);
  float ret2 = __opencl_atomic_fetch_add(d, a, memory_order_relaxed, memory_scope_device);
  float ret3 = __opencl_atomic_fetch_add(d, a, memory_order_relaxed, memory_scope_sub_group);
}
