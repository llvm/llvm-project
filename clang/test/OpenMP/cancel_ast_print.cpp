// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

int main (int argc, char **argv) {
// CHECK: int main(int argc, char **argv) {
#pragma omp parallel
{
#pragma omp cancel parallel if(argc)
}
// CHECK: #pragma omp parallel
// CHECK-NEXT: {
// CHECK-NEXT: #pragma omp cancel parallel if(argc)
// CHECK-NEXT: }
#pragma omp sections
{
#pragma omp cancel sections
}
// CHECK-NEXT: #pragma omp sections
// CHECK: {
// CHECK: #pragma omp cancel sections
// CHECK: }
#pragma omp for
for (int i = 0; i < argc; ++i) {
#pragma omp cancel for if(cancel:argc)
}
// CHECK: #pragma omp for
// CHECK-NEXT: for (int i = 0; i < argc; ++i) {
// CHECK-NEXT: #pragma omp cancel for if(cancel: argc)
// CHECK-NEXT: }
#pragma omp task
{
#pragma omp cancel taskgroup
}
// CHECK: #pragma omp task
// CHECK: {
// CHECK: #pragma omp cancel taskgroup
// CHECK: }
// CHECK: return argc;
  return argc;
}

#endif
