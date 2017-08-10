// RUN: %clang_cc1 < %s -cl-std=CL2.0 -finclude-default-header -triple spir64 -emit-llvm | FileCheck -check-prefix=SPIR %s
// RUN: %clang_cc1 < %s -cl-std=CL2.0 -finclude-default-header -triple armv5e-none-linux-gnueabi -emit-llvm | FileCheck -check-prefix=ARM %s

void f(atomic_int *i, atomic_uint *ui, int cmp) {
  int x;
  // SPIR: {{%[^ ]*}} = call i32 @__opencl_atomic_load_4(i8 addrspace(4)* {{%[0-9]+}}, i32 5, i32 1)
  // ARM: {{%[^ ]*}} = call i32 @__opencl_atomic_load_4(i8* {{%[0-9]+}}, i32 5, i32 1)
  x = __opencl_atomic_load(i, memory_order_seq_cst, memory_scope_work_group);
  // SPIR: call void @__opencl_atomic_store_4(i8 addrspace(4)* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 1)
  // ARM: call void @__opencl_atomic_store_4(i8* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 1)
  __opencl_atomic_store(i, 1, memory_order_seq_cst, memory_scope_work_group);
  // SPIR: {{%[^ ]*}} = call i32 @__opencl_atomic_fetch_add_4(i8 addrspace(4)* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 1)
  // ARM: {{%[^ ]*}} = call i32 @__opencl_atomic_fetch_add_4(i8* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 1)
  x = __opencl_atomic_fetch_add(i, 3, memory_order_seq_cst, memory_scope_work_group);
  // SPIR: {{%[^ ]*}} = call i32 @__opencl_atomic_fetch_min_4(i8 addrspace(4)* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 1)
  // ARM: {{%[^ ]*}} = call i32 @__opencl_atomic_fetch_min_4(i8* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 1)
  x = __opencl_atomic_fetch_min(i, 3, memory_order_seq_cst, memory_scope_work_group);
  // SPIR: {{%[^ ]*}} = call i32 @__opencl_atomic_fetch_umin_4(i8 addrspace(4)* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 1)
  // ARM: {{%[^ ]*}} = call i32 @__opencl_atomic_fetch_umin_4(i8* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 1)
  x = __opencl_atomic_fetch_min(ui, 3, memory_order_seq_cst, memory_scope_work_group);
  // SPIR: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8 addrspace(4)* {{%[0-9]+}}, i8 addrspace(4)* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 5, i32 1)
  // ARM: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8* {{%[0-9]+}}, i8* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 5, i32 1)
  x = __opencl_atomic_compare_exchange_strong(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group);
  // SPIR: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8 addrspace(4)* {{%[0-9]+}}, i8 addrspace(4)* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 5, i32 1)
  // ARM: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8* {{%[0-9]+}}, i8* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 5, i32 1)
  x = __opencl_atomic_compare_exchange_weak(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group);
  // SPIR: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8 addrspace(4)* {{%[0-9]+}}, i8 addrspace(4)* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 5, i32 2)
  // ARM: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8* {{%[0-9]+}}, i8* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 5, i32 2)
  x = __opencl_atomic_compare_exchange_weak(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_device);
  // SPIR: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8 addrspace(4)* {{%[0-9]+}}, i8 addrspace(4)* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 5, i32 3)
  // ARM: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8* {{%[0-9]+}}, i8* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 5, i32 3)
  x = __opencl_atomic_compare_exchange_weak(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_all_svm_devices);
#ifdef cl_khr_subgroups
  // SPIR: {{%[^ ]*}} = call zeroext i1 @__opencl_atomic_compare_exchange_4(i8 addrspace(4)* {{%[0-9]+}}, i8 addrspace(4)* {{%[0-9]+}}, i32 {{%[0-9]+}}, i32 5, i32 5, i32 4)
  x = __opencl_atomic_compare_exchange_weak(i, &cmp, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_sub_group);
#endif
}
