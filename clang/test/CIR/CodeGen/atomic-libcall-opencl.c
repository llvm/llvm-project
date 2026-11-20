// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-atomic-alignment -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-atomic-alignment -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// A 24-byte atomic forces frontend libcalls. OpenCL libcalls use the
// __opencl_atomic_ prefix and take the memory scope after the memory orders.
struct Big {
  int x[6];
};

// CIR-LABEL: @load(
// CIR-DAG: %[[ORDER:.+]] = cir.const #cir.int<5> : !s32i
// CIR-DAG: %[[SCOPE:.+]] = cir.const #cir.int<2> : !s32i
// CIR: cir.call @__opencl_atomic_load(%{{.+}}, %{{.+}}, %{{.+}}, %[[ORDER]], %[[SCOPE]])
// LLVM-LABEL: @load(
// LLVM: call void @__opencl_atomic_load(i64 noundef 24, ptr noundef %{{.+}}, ptr noundef %{{.+}}, i32 noundef 5, i32 noundef 2)
struct Big load(_Atomic(struct Big) *p) {
  return __opencl_atomic_load(p, __ATOMIC_SEQ_CST, __OPENCL_MEMORY_SCOPE_DEVICE);
}

// CIR-LABEL: @store(
// CIR-DAG: %[[ORDER:.+]] = cir.const #cir.int<3> : !s32i
// CIR-DAG: %[[SCOPE:.+]] = cir.const #cir.int<1> : !s32i
// CIR: cir.call @__opencl_atomic_store(%{{.+}}, %{{.+}}, %{{.+}}, %[[ORDER]], %[[SCOPE]])
// LLVM-LABEL: @store(
// LLVM: call void @__opencl_atomic_store(i64 noundef 24, ptr noundef %{{.+}}, ptr noundef %{{.+}}, i32 noundef 3, i32 noundef 1)
void store(_Atomic(struct Big) *p, struct Big v) {
  __opencl_atomic_store(p, v, __ATOMIC_RELEASE, __OPENCL_MEMORY_SCOPE_WORK_GROUP);
}

// CIR-LABEL: @exchange(
// CIR-DAG: %[[SCOPE:.+]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR-DAG: %[[ORDER:.+]] = cir.const #cir.int<4> : !s32i
// CIR: cir.call @__opencl_atomic_exchange(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[ORDER]], %[[SCOPE]])
// LLVM-LABEL: @exchange(
// LLVM: call void @__opencl_atomic_exchange(i64 noundef 24, ptr noundef %{{.+}}, ptr noundef %{{.+}}, ptr noundef %{{.+}}, i32 noundef 4, i32 noundef %{{.+}})
struct Big exchange(_Atomic(struct Big) *p, struct Big v, int scope) {
  return __opencl_atomic_exchange(p, v, __ATOMIC_ACQ_REL, scope);
}

// CIR-LABEL: @cmpxchg_strong(
// CIR-DAG: %[[SUCCESS:.+]] = cir.const #cir.int<4> : !s32i
// CIR-DAG: %[[FAILURE:.+]] = cir.const #cir.int<2> : !s32i
// CIR-DAG: %[[SCOPE:.+]] = cir.const #cir.int<3> : !s32i
// CIR: cir.call @__opencl_atomic_compare_exchange(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[SUCCESS]], %[[FAILURE]], %[[SCOPE]])
// LLVM-LABEL: @cmpxchg_strong(
// LLVM: call zeroext i1 @__opencl_atomic_compare_exchange(i64 noundef 24, ptr noundef %{{.+}}, ptr noundef %{{.+}}, ptr noundef %{{.+}}, i32 noundef 4, i32 noundef 2, i32 noundef 3)
_Bool cmpxchg_strong(_Atomic(struct Big) *p, struct Big *expected,
                     struct Big desired) {
  return __opencl_atomic_compare_exchange_strong(
      p, expected, desired, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE,
      __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
}

// CIR-LABEL: @cmpxchg_weak(
// CIR-DAG: %[[SUCCESS_SLOT:.+]] = cir.alloca "success" {{.*}} : !cir.ptr<!s32i>
// CIR-DAG: %[[FAILURE_SLOT:.+]] = cir.alloca "failure" {{.*}} : !cir.ptr<!s32i>
// CIR-DAG: %[[SCOPE_SLOT:.+]] = cir.alloca "scope" {{.*}} : !cir.ptr<!s32i>
// CIR-DAG: %[[SUCCESS:.+]] = cir.load {{.*}} %[[SUCCESS_SLOT]] : !cir.ptr<!s32i>, !s32i
// CIR-DAG: %[[FAILURE:.+]] = cir.load {{.*}} %[[FAILURE_SLOT]] : !cir.ptr<!s32i>, !s32i
// CIR-DAG: %[[SCOPE:.+]] = cir.load {{.*}} %[[SCOPE_SLOT]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.call @__opencl_atomic_compare_exchange(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[SUCCESS]], %[[FAILURE]], %[[SCOPE]])
// LLVM-LABEL: @cmpxchg_weak(
// LLVM: call zeroext i1 @__opencl_atomic_compare_exchange(i64 noundef 24, ptr noundef %{{.+}}, ptr noundef %{{.+}}, ptr noundef %{{.+}}, i32 noundef %{{.+}}, i32 noundef %{{.+}}, i32 noundef %{{.+}})
_Bool cmpxchg_weak(_Atomic(struct Big) *p, struct Big *expected,
                   struct Big desired, int success, int failure, int scope) {
  return __opencl_atomic_compare_exchange_weak(p, expected, desired, success,
                                               failure, scope);
}
