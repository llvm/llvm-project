// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK: cir.func
void omp_parallel_1() {
// CHECK: omp.parallel {
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: }
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
#pragma omp parallel
{
}
}
// CHECK: cir.func
void omp_parallel_2() {
// CHECK: %[[YVarDecl:.+]] = {{.*}} ["y", init]
// CHECK: omp.parallel {
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[XVarDecl:.+]] = {{.*}} ["x", init]
// CHECK-NEXT: %[[C1:.+]] = cir.const(#cir.int<1> : !s32i)
// CHECK-NEXT: cir.store %[[C1]], %[[XVarDecl]]
// CHECK-NEXT: %[[XVal:.+]] = cir.load %[[XVarDecl]]
// CHECK-NEXT: %[[COne:.+]] = cir.const(#cir.int<1> : !s32i)
// CHECK-NEXT: %[[BinOpVal:.+]] = cir.binop(add, %[[XVal]], %[[COne]])
// CHECK-NEXT: cir.store %[[BinOpVal]], %[[YVarDecl]]
// CHECK-NEXT: }
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
  int y = 0;
#pragma omp parallel
{
  int x = 1;
  y = x + 1;
}
}
