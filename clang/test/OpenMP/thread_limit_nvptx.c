// Test target codegen - host bc file has to be created first.
//
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

void foo(int N) {
// CHECK: l11, !"maxntidx", i32 128}
#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < N; ++i)
    ;
// CHECK: l15, !"maxntidx", i32 4}
#pragma omp target teams distribute parallel for simd thread_limit(4)
  for (int i = 0; i < N; ++i)
    ;
// CHECK-NOT: l21, !"maxntidx", i32 128}
// CHECK: l21, !"maxntidx", i32 42}
// CHECK-NOT: l21, !"maxntidx", i32 128}
#pragma omp target teams distribute parallel for simd ompx_attribute(__attribute__((launch_bounds(42, 42))))
  for (int i = 0; i < N; ++i)
    ;
// CHECK-NOT: l27, !"maxntidx", i32 42}
// CHECK: l27, !"maxntidx", i32 22}
// CHECK-NOT: l27, !"maxntidx", i32 42}
#pragma omp target teams distribute parallel for simd ompx_attribute(__attribute__((launch_bounds(42, 42)))) num_threads(22)
  for (int i = 0; i < N; ++i)
    ;
}

#endif

