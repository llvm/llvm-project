// RUN: %clang_cc1 -no-opaque-pointers < %s -cl-std=CL2.0 -triple spir64 -emit-llvm | FileCheck -check-prefix=SPIR %s
// RUN: %clang_cc1 -no-opaque-pointers < %s -cl-std=CL2.0 -triple armv5e-none-linux-gnueabi -emit-llvm | FileCheck -check-prefix=ARM %s
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
  // SPIR: {{%[^ ]*}} = call i32 @__opencl_atomic_load_4(i8 addrspace(4)* noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  // ARM: {{%[^ ]*}} = call i32 @__opencl_atomic_load_4(i8* noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  x = __opencl_atomic_load(i, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: call void @__opencl_atomic_store_4(i8 addrspace(4)* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  // ARM: call void @__opencl_atomic_store_4(i8* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  __opencl_atomic_store(i, 1, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: %[[GP:[0-9]+]] = addrspacecast i8 addrspace(1)* {{%[0-9]+}} to i8 addrspace(4)*
  // SPIR: call void @__opencl_atomic_store_4(i8 addrspace(4)* noundef %[[GP]], i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  // ARM: call void @__opencl_atomic_store_4(i8* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  __opencl_atomic_store(gi, 1, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: %[[GP:[0-9]+]] = addrspacecast i8 addrspace(3)* {{%[0-9]+}} to i8 addrspace(4)*
  // SPIR: call void @__opencl_atomic_store_4(i8 addrspace(4)* noundef %[[GP]], i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  // ARM: call void @__opencl_atomic_store_4(i8* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  __opencl_atomic_store(li, 1, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: %[[GP:[0-9]+]] = addrspacecast i8* {{%[0-9]+}} to i8 addrspace(4)*
  // SPIR: call void @__opencl_atomic_store_4(i8 addrspace(4)* noundef %[[GP]], i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  // ARM: call void @__opencl_atomic_store_4(i8* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  __opencl_atomic_store(pi, 1, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: {{%[^ ]*}} = call i32 @__opencl_atomic_fetch_add_4(i8 addrspace(4)* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  // ARM: {{%[^ ]*}} = call i32 @__opencl_atomic_fetch_add_4(i8* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  x = __opencl_atomic_fetch_add(i, 3, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: {{%[^ ]*}} = call i32 @__opencl_atomic_fetch_min_4(i8 addrspace(4)* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  // ARM: {{%[^ ]*}} = call i32 @__opencl_atomic_fetch_min_4(i8* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  x = __opencl_atomic_fetch_min(i, 3, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: {{%[^ ]*}} = call i32 @__opencl_atomic_fetch_umin_4(i8 addrspace(4)* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  // ARM: {{%[^ ]*}} = call i32 @__opencl_atomic_fetch_umin_4(i8* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 1)
  x = __opencl_atomic_fetch_min(ui, 3, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8 addrspace(4)* noundef {{%[0-9]+}}, i8 addrspace(4)* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 5, i32 noundef 1)
  // ARM: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8* noundef {{%[0-9]+}}, i8* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 5, i32 noundef 1)
  x = __opencl_atomic_compare_exchange_strong(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8 addrspace(4)* noundef {{%[0-9]+}}, i8 addrspace(4)* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 5, i32 noundef 1)
  // ARM: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8* noundef {{%[0-9]+}}, i8* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 5, i32 noundef 1)
  x = __opencl_atomic_compare_exchange_weak(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group);

  // SPIR: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8 addrspace(4)* noundef {{%[0-9]+}}, i8 addrspace(4)* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 5, i32 noundef 2)
  // ARM: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8* noundef {{%[0-9]+}}, i8* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 5, i32 noundef 2)
  x = __opencl_atomic_compare_exchange_weak(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_device);

  // SPIR: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8 addrspace(4)* noundef {{%[0-9]+}}, i8 addrspace(4)* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 5, i32 noundef 3)
  // ARM: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8* noundef {{%[0-9]+}}, i8* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 5, i32 noundef 3)
  x = __opencl_atomic_compare_exchange_weak(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_all_svm_devices);

#ifdef cl_khr_subgroups
  // SPIR: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8 addrspace(4)* noundef {{%[0-9]+}}, i8 addrspace(4)* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 5, i32 noundef 4)
  x = __opencl_atomic_compare_exchange_weak(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_sub_group);
#endif

  // SPIR: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8 addrspace(4)* noundef {{%[0-9]+}}, i8 addrspace(4)* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}})
  // ARM: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8* noundef {{%[0-9]+}}, i8* noundef {{%[0-9]+}}, i32 noundef {{%[0-9]+}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}})
  x = __opencl_atomic_compare_exchange_weak(i, &cmp, 1, order, order, scope);
}
