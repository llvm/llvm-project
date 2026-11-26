// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

void signed_loop(float *V, int N) {
#pragma omp teams distribute parallel for schedule(static, 1)
  for (int i = 0; i < N; ++i)
    ++V[i];
}

// CHECK-LABEL: define internal void @signed_loop.omp_outlined.omp_outlined(
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: omp.inner.for.cond:
// CHECK-NEXT: [[IV:%.*]] = load i32, ptr %.omp.iv, align 4
// CHECK-NEXT: [[PREV_UB:%.*]] = load i64, ptr %.previous.ub..addr, align 8
// CHECK-NEXT: [[BOUND:%.*]] = trunc i64 [[PREV_UB]] to i32
// CHECK-NEXT: [[CMP:%.*]] = icmp sle i32 [[IV]], [[BOUND]]

void unsigned_loop(float *V, unsigned N) {
#pragma omp teams distribute parallel for schedule(static, 1)
  for (unsigned i = 0; i < N; ++i)
    ++V[i];
}

// CHECK-LABEL: define internal void @unsigned_loop.omp_outlined.omp_outlined(
// CHECK: call void @__kmpc_for_static_init_4u(
// CHECK: omp.inner.for.cond:
// CHECK-NEXT: [[IV:%.*]] = load i32, ptr %.omp.iv, align 4
// CHECK-NEXT: [[PREV_UB:%.*]] = load i64, ptr %.previous.ub..addr, align 8
// CHECK-NEXT: [[BOUND:%.*]] = trunc i64 [[PREV_UB]] to i32
// CHECK-NEXT: [[NEXT_BOUND:%.*]] = add i32 [[BOUND]], 1
// CHECK-NEXT: [[CMP:%.*]] = icmp ult i32 [[IV]], [[NEXT_BOUND]]
