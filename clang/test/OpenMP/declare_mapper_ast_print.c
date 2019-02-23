// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK: struct vec {
struct vec {
  int len;
  double *data;
};
// CHECK: };

// CHECK: struct dat {
struct dat {
  int i;
  double d;
#pragma omp declare mapper(id: struct vec v) map(v.len)
// CHECK: #pragma omp declare mapper (id : struct vec v) map(tofrom: v.len){{$}}
};
// CHECK: };

#pragma omp declare mapper(id: struct vec v) map(v.len)
// CHECK: #pragma omp declare mapper (id : struct vec v) map(tofrom: v.len){{$}}
#pragma omp declare mapper(default : struct vec kk) map(kk.len) map(kk.data[0:2])
// CHECK: #pragma omp declare mapper (default : struct vec kk) map(tofrom: kk.len) map(tofrom: kk.data[0:2]){{$}}
#pragma omp declare mapper(struct dat d) map(to: d.d)
// CHECK: #pragma omp declare mapper (default : struct dat d) map(to: d.d){{$}}

// CHECK: int main() {
int main() {
#pragma omp declare mapper(id: struct vec v) map(v.len)
// CHECK: #pragma omp declare mapper (id : struct vec v) map(tofrom: v.len)
  {
#pragma omp declare mapper(id: struct vec v) map(v.len)
// CHECK: #pragma omp declare mapper (id : struct vec v) map(tofrom: v.len)
    struct vec vv;
    struct dat dd[10];
#pragma omp target map(mapper(id) alloc: vv)
// CHECK: #pragma omp target map(mapper(id),alloc: vv)
    { vv.len++; }
#pragma omp target map(mapper(default), from: dd[0:10])
// CHECK: #pragma omp target map(mapper(default),from: dd[0:10])
    { dd[0].i++; }
#pragma omp target update to(mapper(id): vv)
// CHECK: #pragma omp target update to(mapper(id): vv)
  }
  return 0;
}
// CHECK: }

#endif
