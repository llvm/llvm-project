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

// rdar://13489679

extern _Atomic(_Bool) a_bool;
extern _Atomic(float) a_float;
extern _Atomic(void*) a_pointer;
extern _Atomic(pointer_pair_t) a_pointer_pair;
extern _Atomic(pointer_quad_t) a_pointer_quad;

// CHECK-LABEL:define{{.*}} void @test0()
// CHECK:      [[TEMP:%.*]] = alloca i8, align 1
// CHECK-NEXT: store i8 1, i8* [[TEMP]]
// CHECK-NEXT: [[T0:%.*]] = load i8, i8* [[TEMP]], align 1
// CHECK-NEXT: store atomic i8 [[T0]], i8* @a_bool seq_cst, align 1
void test0() {
  __c11_atomic_store(&a_bool, 1, memory_order_seq_cst);
}

// CHECK-LABEL:define{{.*}} void @test1()
// CHECK:      [[TEMP:%.*]] = alloca float, align 4
// CHECK-NEXT: store float 3.000000e+00, float* [[TEMP]]
// CHECK-NEXT: [[T0:%.*]] = bitcast float* [[TEMP]] to i32*
// CHECK-NEXT: [[T1:%.*]] = load i32, i32* [[T0]], align 4
// CHECK-NEXT: store atomic i32 [[T1]], i32* bitcast (float* @a_float to i32*) seq_cst, align 4
void test1() {
  __c11_atomic_store(&a_float, 3, memory_order_seq_cst);
}

// CHECK-LABEL:define{{.*}} void @test2()
// CHECK:      [[TEMP:%.*]] = alloca i8*, align 8
// CHECK-NEXT: store i8* @a_bool, i8** [[TEMP]]
// CHECK-NEXT: [[T0:%.*]] = bitcast i8** [[TEMP]] to i64*
// CHECK-NEXT: [[T1:%.*]] = load i64, i64* [[T0]], align 8
// CHECK-NEXT: store atomic i64 [[T1]], i64* bitcast (i8** @a_pointer to i64*) seq_cst, align 8
void test2() {
  __c11_atomic_store(&a_pointer, &a_bool, memory_order_seq_cst);
}

// CHECK-LABEL:define{{.*}} void @test3(
// CHECK:      [[PAIR:%.*]] = alloca [[PAIR_T:%.*]], align 8
// CHECK-NEXT: [[TEMP:%.*]] = alloca [[PAIR_T]], align 8
// CHECK:      llvm.memcpy
// CHECK-NEXT: [[T0:%.*]] = bitcast [[PAIR_T]]* [[TEMP]] to i128*
// CHECK-NEXT: [[T1:%.*]] = load i128, i128* [[T0]], align 8
// CHECK-NEXT: store atomic i128 [[T1]], i128* bitcast ([[PAIR_T]]* @a_pointer_pair to i128*) seq_cst, align 16
void test3(pointer_pair_t pair) {
  __c11_atomic_store(&a_pointer_pair, pair, memory_order_seq_cst);
}

// CHECK-LABEL:define{{.*}} void @test4(
// CHECK:      [[TEMP:%.*]] = alloca [[QUAD_T:%.*]], align 8
// CHECK-NEXT: [[T0:%.*]] = bitcast [[QUAD_T]]* [[TEMP]] to i8*
// CHECK-NEXT: [[T1:%.*]] = bitcast [[QUAD_T]]* {{%.*}} to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[T0]], i8* align 8 [[T1]], i64 32, i1 false)
// CHECK-NEXT: [[T0:%.*]] = bitcast [[QUAD_T]]* [[TEMP]] to i256*
// CHECK-NEXT: [[T1:%.*]] = bitcast i256* [[T0]] to i8*
// CHECK-NEXT: call void @__atomic_store(i64 noundef 32, i8* noundef bitcast ([[QUAD_T]]* @a_pointer_quad to i8*), i8* noundef [[T1]], i32 noundef 5)
void test4(pointer_quad_t quad) {
  __c11_atomic_store(&a_pointer_quad, quad, memory_order_seq_cst);
}
