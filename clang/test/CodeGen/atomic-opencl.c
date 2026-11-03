// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -o - | FileCheck %s

// A 24-byte atomic forces frontend libcalls. OpenCL libcalls use the
// __opencl_atomic_ prefix and take the memory scope after the memory orders.
struct Big {
  int x[6];
};

// CHECK-LABEL: @load(
// CHECK: call void @__opencl_atomic_load(i64 noundef 24, ptr noundef %{{.+}}, ptr noundef %{{.+}}, i32 noundef 5, i32 noundef 2)
struct Big load(_Atomic(struct Big) *p) {
  return __opencl_atomic_load(p, __ATOMIC_SEQ_CST, __OPENCL_MEMORY_SCOPE_DEVICE);
}

// CHECK-LABEL: @store(
// CHECK: call void @__opencl_atomic_store(i64 noundef 24, ptr noundef %{{.+}}, ptr noundef %{{.+}}, i32 noundef 3, i32 noundef 1)
void store(_Atomic(struct Big) *p, struct Big v) {
  __opencl_atomic_store(p, v, __ATOMIC_RELEASE, __OPENCL_MEMORY_SCOPE_WORK_GROUP);
}

// CHECK-LABEL: @exchange(
// CHECK: call void @__opencl_atomic_exchange(i64 noundef 24, ptr noundef %{{.+}}, ptr noundef %{{.+}}, ptr noundef %{{.+}}, i32 noundef 4, i32 noundef %{{.+}})
struct Big exchange(_Atomic(struct Big) *p, struct Big v, int scope) {
  return __opencl_atomic_exchange(p, v, __ATOMIC_ACQ_REL, scope);
}

// CHECK-LABEL: @cmpxchg_strong(
// CHECK: call zeroext i1 @__opencl_atomic_compare_exchange(i64 noundef 24, ptr noundef %{{.+}}, ptr noundef %{{.+}}, ptr noundef %{{.+}}, i32 noundef 4, i32 noundef 2, i32 noundef 3)
_Bool cmpxchg_strong(_Atomic(struct Big) *p, struct Big *expected,
                     struct Big desired) {
  return __opencl_atomic_compare_exchange_strong(
      p, expected, desired, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE,
      __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
}

// CHECK-LABEL: @cmpxchg_weak(
// CHECK: call zeroext i1 @__opencl_atomic_compare_exchange(i64 noundef 24, ptr noundef %{{.+}}, ptr noundef %{{.+}}, ptr noundef %{{.+}}, i32 noundef %{{.+}}, i32 noundef %{{.+}}, i32 noundef %{{.+}})
_Bool cmpxchg_weak(_Atomic(struct Big) *p, struct Big *expected,
                   struct Big desired, int success, int failure, int scope) {
  return __opencl_atomic_compare_exchange_weak(p, expected, desired, success,
                                               failure, scope);
}
