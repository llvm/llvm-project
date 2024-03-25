// RUN: %clang_cc1 %s -emit-llvm -o - -triple=arm64-apple-ios7 | FileCheck %s

// Memory ordering values.
enum {
  memory_order_relaxed = 0,
  memory_order_consume = 1,
  memory_order_acquire = 2,
  memory_order_release = 3,
  memory_order_acq_rel = 4,
  memory_order_seq_cst = 5
};

typedef struct { void *a, *b; } pointer_pair_t;
typedef struct { void *a, *b, *c, *d; } pointer_quad_t;

extern _Atomic(_Bool) a_bool;
extern _Atomic(float) a_float;
extern _Atomic(void*) a_pointer;
extern _Atomic(pointer_pair_t) a_pointer_pair;
extern _Atomic(pointer_quad_t) a_pointer_quad;

// CHECK-LABEL:define{{.*}} void @test0()
// CHECK:      [[TEMP:%.*]] = alloca i8, align 1
// CHECK-NEXT: store i8 1, ptr [[TEMP]]
// CHECK-NEXT: [[T0:%.*]] = load i8, ptr [[TEMP]], align 1
// CHECK-NEXT: store atomic i8 [[T0]], ptr @a_bool seq_cst, align 1
void test0(void) {
  __c11_atomic_store(&a_bool, 1, memory_order_seq_cst);
}

// CHECK-LABEL:define{{.*}} void @test1()
// CHECK:      [[TEMP:%.*]] = alloca float, align 4
// CHECK-NEXT: store float 3.000000e+00, ptr [[TEMP]]
// CHECK-NEXT: [[T1:%.*]] = load i32, ptr [[TEMP]], align 4
// CHECK-NEXT: store atomic i32 [[T1]], ptr @a_float seq_cst, align 4
void test1(void) {
  __c11_atomic_store(&a_float, 3, memory_order_seq_cst);
}

// CHECK-LABEL:define{{.*}} void @test2()
// CHECK:      [[TEMP:%.*]] = alloca ptr, align 8
// CHECK-NEXT: store ptr @a_bool, ptr [[TEMP]]
// CHECK-NEXT: [[T1:%.*]] = load i64, ptr [[TEMP]], align 8
// CHECK-NEXT: store atomic i64 [[T1]], ptr @a_pointer seq_cst, align 8
void test2(void) {
  __c11_atomic_store(&a_pointer, &a_bool, memory_order_seq_cst);
}

// CHECK-LABEL:define{{.*}} void @test3(
// CHECK:      [[PAIR:%.*]] = alloca [[PAIR_T:%.*]], align 8
// CHECK-NEXT: [[TEMP:%.*]] = alloca [[PAIR_T]], align 8
// CHECK:      llvm.memcpy
// CHECK-NEXT: [[T1:%.*]] = load i128, ptr [[TEMP]], align 8
// CHECK-NEXT: store atomic i128 [[T1]], ptr @a_pointer_pair seq_cst, align 16
void test3(pointer_pair_t pair) {
  __c11_atomic_store(&a_pointer_pair, pair, memory_order_seq_cst);
}

// CHECK-LABEL:define{{.*}} void @test4(
// CHECK-SAME: ptr noundef [[QUAD:%.*]])
// CHECK:      [[QUAD_INDIRECT_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT: [[TEMP:%.*]] = alloca [[QUAD_T:%.*]], align 8
// CHECK-NEXT: store ptr [[QUAD]], ptr [[QUAD_INDIRECT_ADDR]]
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[TEMP]], ptr align 8 {{%.*}}, i64 32, i1 false)
// CHECK-NEXT: call void @__atomic_store(i64 noundef 32, ptr noundef @a_pointer_quad, ptr noundef [[TEMP]], i32 noundef 5)
void test4(pointer_quad_t quad) {
  __c11_atomic_store(&a_pointer_quad, quad, memory_order_seq_cst);
}
