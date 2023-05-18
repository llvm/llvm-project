// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm -o - %s | \
// RUN:     FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm -o - %s | \
// RUN:     FileCheck %s

void foo(void) {
  #pragma omp target
  ;
}
