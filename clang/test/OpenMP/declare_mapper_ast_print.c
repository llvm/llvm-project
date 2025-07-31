// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -include-pch %t -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -include-pch %t -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -DOMP51 -verify -fopenmp -ast-print %s | FileCheck -check-prefixes=CHECK,OMP51 %s
// RUN: %clang_cc1 -DOMP51 -fopenmp -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP51 -fopenmp -include-pch %t -verify %s -ast-print | FileCheck -check-prefixes=CHECK,OMP51 %s

// RUN: %clang_cc1 -DOMP51 -verify -fopenmp-simd -ast-print %s | FileCheck -check-prefixes=CHECK,OMP51 %s
// RUN: %clang_cc1 -DOMP51 -fopenmp-simd -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP51 -fopenmp-simd -include-pch %t -verify %s -ast-print | FileCheck -check-prefixes=CHECK,OMP51 %s

// RUN: %clang_cc1 -DOMP52 -verify -fopenmp -fopenmp-version=52 -ast-print %s | FileCheck -check-prefixes=CHECK,OMP52 %s
// RUN: %clang_cc1 -DOMP52 -fopenmp -fopenmp-version=52 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP52 -fopenmp -fopenmp-version=52 -include-pch %t -verify %s -ast-print | FileCheck -check-prefixes=CHECK,OMP52 %s

// RUN: %clang_cc1 -DOMP52 -verify -fopenmp-simd -fopenmp-version=52 -ast-print %s | FileCheck -check-prefixes=CHECK,OMP52 %s
// RUN: %clang_cc1 -DOMP52 -fopenmp-simd -fopenmp-version=52 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP52 -fopenmp-simd -fopenmp-version=52 -include-pch %t -verify %s -ast-print | FileCheck -check-prefixes=CHECK,OMP52 %s
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

// Verify that nested default mappers do not lead to a crash during parsing / sema.
// CHECK: struct inner {
struct inner {
  int size;
  int *data;
};
#pragma omp declare mapper(struct inner i) map(i, i.data[0 : i.size])
// CHECK: #pragma omp declare mapper (default : struct inner i) map(tofrom: default::i,i.data[0:i.size]){{$}}

// CHECK: struct outer {
struct outer {
  int a;
  struct inner i;
};
#pragma omp declare mapper(struct outer o) map(o)
// CHECK: #pragma omp declare mapper (default : struct outer o) map(tofrom: default::o) map(tofrom: o.i){{$}}

// CHECK: int main(void) {
int main(void) {
#pragma omp declare mapper(id: struct vec v) map(v.len)
// CHECK: #pragma omp declare mapper (id : struct vec v) map(tofrom: v.len)
  {
#pragma omp declare mapper(id: struct vec v) map(v.len)
// CHECK: #pragma omp declare mapper (id : struct vec v) map(tofrom: v.len)
    struct vec vv;
    struct dat dd[10];
#pragma omp target map(mapper(id), alloc: vv)
// CHECK: #pragma omp target map(mapper(id),alloc: vv)
    { vv.len++; }
#pragma omp target map(mapper(default), from: dd[0:10])
// CHECK: #pragma omp target map(mapper(default),from: dd[0:10])
    { dd[0].i++; }
#pragma omp target update to(mapper(id): vv) from(mapper(default): dd[0:10])
// CHECK: #pragma omp target update to(mapper(id): vv) from(mapper(default): dd[0:10])
#ifdef OMP51
#pragma omp target update to(mapper(id) present: vv) from(mapper(default), present: dd[0:10])
// OMP51: #pragma omp target update to(mapper(id), present: vv) from(mapper(default), present: dd[0:10])
#pragma omp target update to(present mapper(id): vv) from(present, mapper(default): dd[0:10])
// OMP51: #pragma omp target update to(present, mapper(id): vv) from(present, mapper(default): dd[0:10])
#endif
  }
#ifdef OMP52
#pragma omp declare mapper(id1: struct vec vvec) map(iterator(it=0:vvec.len:2), tofrom:vvec.data[it])
// OMP52: #pragma omp declare mapper (id1 : struct vec vvec) map(iterator(int it = 0:vvec.len:2),tofrom: vvec.data[it]);
#endif

  {
    struct outer outer;
#pragma omp target map(outer)
// CHECK: #pragma omp target map(tofrom: outer)
    { }
  }

  return 0;
}
// CHECK: }

#endif
