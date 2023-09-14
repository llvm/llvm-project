// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm -fexceptions -fcxx-exceptions -o - %s | FileCheck %s

// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm -fexceptions -fcxx-exceptions -o - %s | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

int main() {
// CHECK: call i32 @__kmpc_global_thread_num
// CHECK: call void @__kmpc_taskgraph
// CHECK: @taskgraph.omp_outlined.
#pragma ompx taskgraph
{}
// CHECK: call void @__kmpc_taskgraph
// CHECK: @taskgraph.omp_outlined..1
  int foo = 0;
#pragma ompx taskgraph
{
  foo++;
}
// CHECK: call void @__kmpc_taskgraph
// CHECK: @taskgraph.omp_outlined..2
  for(int i = 0; i < 10; ++i)
#pragma ompx taskgraph
{
  #pragma omp task
    foo++;
}
  return 0;
}

