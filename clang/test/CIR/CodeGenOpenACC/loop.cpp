// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

extern "C" void acc_loop(int *A, int *B, int *C, int N) {
  // CHECK: cir.func @acc_loop(%[[ARG_A:.*]]: !cir.ptr<!s32i> loc{{.*}}, %[[ARG_B:.*]]: !cir.ptr<!s32i> loc{{.*}}, %[[ARG_C:.*]]: !cir.ptr<!s32i> loc{{.*}}, %[[ARG_N:.*]]: !s32i loc{{.*}}) {
  // CHECK-NEXT: %[[ALLOCA_A:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["A", init]
  // CHECK-NEXT: %[[ALLOCA_B:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["B", init]
  // CHECK-NEXT: %[[ALLOCA_C:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["C", init]
  // CHECK-NEXT: %[[ALLOCA_N:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["N", init]
  // CHECK-NEXT: cir.store %[[ARG_A]], %[[ALLOCA_A]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK-NEXT: cir.store %[[ARG_B]], %[[ALLOCA_B]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK-NEXT: cir.store %[[ARG_C]], %[[ALLOCA_C]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK-NEXT: cir.store %[[ARG_N]], %[[ALLOCA_N]] : !s32i, !cir.ptr<!s32i>


#pragma acc loop
  for (unsigned I = 0u; I < N; ++I) {
    A[I] = B[I] + C[I];
  }
  // CHECK-NEXT: acc.loop {
  // CHECK-NEXT: cir.scope {
  // CHECK: cir.for : cond {
  // CHECK: cir.condition
  // CHECK-NEXT: } body {
  // CHECK-NEXT: cir.scope {
  // CHECK: }
  // CHECK-NEXT: cir.yield
  // CHECK-NEXT: } step {
  // CHECK: cir.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
}
