// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

extern "C" void acc_combined(int N) {
  // CHECK: cir.func @acc_combined(%[[ARG_N:.*]]: !s32i loc{{.*}}) {
  // CHECK-NEXT: %[[ALLOCA_N:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["N", init]
  // CHECK-NEXT: cir.store %[[ARG_N]], %[[ALLOCA_N]] : !s32i, !cir.ptr<!s32i>

#pragma acc parallel loop
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.serial combined(loop) {
  // CHECK: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
#pragma acc kernels loop
  for(unsigned I = 0; I < N; ++I);

  // CHECK: acc.kernels combined(loop) {
  // CHECK: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc
}
