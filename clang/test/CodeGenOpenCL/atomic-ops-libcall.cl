// RUN: %clang_cc1 < %s -cl-std=CL2.0 -triple spir64 -emit-llvm | FileCheck -check-prefix=SPIR %s
// RUN: %clang_cc1 < %s -cl-std=CL2.0 -triple armv5e-none-linux-gnueabi -emit-llvm | FileCheck -check-prefix=ARM %s
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

void f(atomic_int *i, global atomic_int *gi, local atomic_int *li, private atomic_int *pi, atomic_uint *ui, int cmp, int order, int scope) {
  int x;
  // SPIR: load atomic i32, ptr addrspace(4) {{.*}} seq_cst, align 4
  // ARM: load atomic i32, ptr {{.*}} seq_cst, align 4
  x = __opencl_atomic_load(i, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: store atomic i32 {{.*}}, ptr addrspace(4) {{.*}} seq_cst, align 4
  // ARM: store atomic i32 {{.*}}, ptr {{.*}} seq_cst, align 4
  __opencl_atomic_store(i, 1, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: store atomic i32 {{.*}}, ptr addrspace(1) {{.*}} seq_cst, align 4
  // ARM: store atomic i32 {{.*}}, ptr {{.*}} seq_cst, align 4
  __opencl_atomic_store(gi, 1, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: store atomic i32 {{.*}}, ptr addrspace(3) {{.*}} seq_cst, align 4
  // ARM: store atomic i32 {{.*}}, ptr {{.*}} seq_cst, align 4
  __opencl_atomic_store(li, 1, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: store atomic i32 {{.*}}, ptr {{.*}} seq_cst, align 4
  // ARM: store atomic i32 {{.*}}, ptr {{.*}} seq_cst, align 4
  __opencl_atomic_store(pi, 1, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: atomicrmw add ptr addrspace(4) {{.*}}, i32 {{.*}} seq_cst, align 4
  // ARM: atomicrmw add ptr {{.*}}, i32 {{.*}} seq_cst, align 4
  x = __opencl_atomic_fetch_add(i, 3, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: atomicrmw min ptr addrspace(4) {{.*}}, i32 {{.*}} seq_cst, align 4
  // ARM: atomicrmw min ptr {{.*}}, i32 {{.*}} seq_cst, align 4
  x = __opencl_atomic_fetch_min(i, 3, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: atomicrmw umin ptr addrspace(4) {{.*}}, i32 {{.*}} seq_cst, align 4
  // ARM: atomicrmw umin ptr {{.*}}, i32 {{.*}} seq_cst, align 4
  x = __opencl_atomic_fetch_min(ui, 3, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: cmpxchg ptr addrspace(4) {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4
  // ARM: cmpxchg ptr {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4
  x = __opencl_atomic_compare_exchange_strong(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: cmpxchg weak ptr addrspace(4) {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4
  // ARM: cmpxchg weak ptr {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4
  x = __opencl_atomic_compare_exchange_weak(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: cmpxchg weak ptr addrspace(4) {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4
  // ARM: cmpxchg weak ptr {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4
  x = __opencl_atomic_compare_exchange_weak(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_device);

  // SPIR: cmpxchg weak ptr addrspace(4) {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4
  // ARM: cmpxchg weak ptr {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4
  x = __opencl_atomic_compare_exchange_weak(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_all_svm_devices);

#ifdef cl_khr_subgroups
  // SPIR: cmpxchg weak ptr addrspace(4) {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4
  x = __opencl_atomic_compare_exchange_weak(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_sub_group);
#endif

  // SPIR: cmpxchg weak ptr addrspace(4) {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4
  // ARM: cmpxchg weak ptr {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4
  x = __opencl_atomic_compare_exchange_weak(i, &cmp, 1, order, order, scope);
}
