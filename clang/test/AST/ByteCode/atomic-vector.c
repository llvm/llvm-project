// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -o -                                         | FileCheck %s
// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -o - -fexperimental-new-constant-interpreter | FileCheck %s

typedef int A __attribute__((__vector_size__(16)));

void vector_to_atomic_vector() {
  _Atomic(A) atomic_vec;
  A vec;
  atomic_vec = vec;
}

// CHECK: %[[ATOMIC_VEC_ADDR:.*]] = alloca <4 x i32>, align 16
// CHECK: %[[VEC_ADDR:.*]] = alloca <4 x i32>, align 16
// CHECK: %[[ATOMIC_TMP:.*]] = alloca <4 x i32>, align 16
// CHECK: %[[TMP_VEC:.*]] = load <4 x i32>, ptr %[[VEC_ADDR]], align 16
// CHECK: store <4 x i32> %[[TMP_VEC]], ptr %[[ATOMIC_TMP]], align 16
// CHECK: call void @__atomic_store(i64 noundef 16, ptr noundef %[[ATOMIC_VEC_ADDR]], ptr noundef %[[ATOMIC_TMP]], i32 noundef 5)
