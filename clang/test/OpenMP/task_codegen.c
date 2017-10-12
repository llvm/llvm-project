// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

void foo();

// CHECK-LABEL: @main
int main() {
  // CHECK: call i32 @__kmpc_global_thread_num(
  // CHECK: call i8* @__kmpc_omp_task_alloc(
  // CHECK: call i32 @__kmpc_omp_task(
#pragma omp task
  {
#pragma omp taskgroup
    {
#pragma omp task
      foo();
    }
  }
  // CHECK: ret i32 0
  return 0;
}
// CHECK: call void @__kmpc_taskgroup(
// CHECK: call i8* @__kmpc_omp_task_alloc(
// CHECK: call i32 @__kmpc_omp_task(
// CHECK: call void @__kmpc_end_taskgroup(
#endif
