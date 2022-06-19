// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -fopenmp-version=50 %s -triple x86_64-apple-darwin10 -x c++ -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s -triple x86_64-apple-darwin10
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -std=c++11 -include-pch %t -verify %s -triple x86_64-apple-darwin10 -x c++ -emit-llvm -o -| FileCheck %s

// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd %s -fopenmp-version=50 -x c++ -emit-llvm -triple x86_64-apple-darwin10 -o -| FileCheck %s --check-prefix SIMD-ONLY0
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s -triple x86_64-apple-darwin10
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=50 -std=c++11 -include-pch %t -verify %s -emit-llvm -x c++ -emit-llvm -triple x86_64-apple-darwin10 -o -| FileCheck %s --check-prefix SIMD-ONLY0
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#pragma omp requires atomic_default_mem_order(seq_cst)

// CHECK-LABEL: foo
void foo() {
  int a = 0, b = 0;
// CHECK: load atomic i32, {{.*}} seq_cst, align 4
#pragma omp atomic read
  a = b;
// CHECK: store atomic i32 {{.*}} seq_cst, align 4
#pragma omp atomic write
  a = b;
// CHECK: atomicrmw add i32* {{.*}} seq_cst, align 4
#pragma omp atomic
  a += 1;
// CHECK: atomicrmw add i32* {{.*}} seq_cst, align 4
#pragma omp atomic update
  a += 1;
// CHECK: atomicrmw add i32* {{.*}} seq_cst, align 4
#pragma omp atomic capture
  {
    b = a;
    a += 1;
  }
}

#endif
