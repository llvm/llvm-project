// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK: namespace N1 {
namespace N1
{
// CHECK: class vec {
class vec {
public:
  int len;
  double *data;
};
// CHECK: };

// CHECK: class vecchild : public N1::vec {
class vecchild : public vec {
public:
  int lenc;
};
// CHECK: };

#pragma omp declare mapper(id: vec v) map(v.len)
// CHECK: #pragma omp declare mapper (id : N1::vec v) map(tofrom: v.len){{$}}
};
// CHECK: }
// CHECK: ;

template <class T>
class dat {
public:
  class datin {
  public:
    T in;
  };
  int i;
  T d;
#pragma omp declare mapper(id: N1::vec v) map(v.len)
#pragma omp declare mapper(id: datin v) map(v.in)
};

// CHECK: template <class T> class dat {
// CHECK: #pragma omp declare mapper (id : N1::vec v) map(tofrom: v.len){{$}}
// CHECK: #pragma omp declare mapper (id : dat::datin v) map(tofrom: v.in){{$}}
// CHECK: };
// CHECK: template<> class dat<double> {
// CHECK: #pragma omp declare mapper (id : N1::vec v) map(tofrom: v.len){{$}}
// CHECK: #pragma omp declare mapper (id : dat<double>::datin v) map(tofrom: v.in){{$}}
// CHECK: };

#pragma omp declare mapper(default : N1::vec kk) map(kk.len) map(kk.data[0:2])
// CHECK: #pragma omp declare mapper (default : N1::vec kk) map(tofrom: kk.len) map(tofrom: kk.data[0:2]){{$}}
#pragma omp declare mapper(dat<double> d) map(to: d.d)
// CHECK: #pragma omp declare mapper (default : dat<double> d) map(to: d.d){{$}}

template <typename T>
T foo(T a) {
  struct foodatchild {
    T k;
  };
  struct foodat {
    T a;
    struct foodatchild b;
  };
#pragma omp declare mapper(id: struct foodat v) map(v.a)
#pragma omp declare mapper(idd: struct foodatchild v) map(v.k)
#pragma omp declare mapper(id: N1::vec v) map(v.len)
  {
#pragma omp declare mapper(id: N1::vec v) map(v.len)
  }
  struct foodat fd;
#pragma omp target map(mapper(id) alloc: fd)
  { fd.a++; }
#pragma omp target map(mapper(idd) alloc: fd.b)
  { fd.b.k++; }
#pragma omp target update to(mapper(id): fd)
#pragma omp target update to(mapper(idd): fd.b)
#pragma omp target update from(mapper(id): fd)
#pragma omp target update from(mapper(idd): fd.b)
  return 0;
}

// CHECK: template <typename T> T foo(T a) {
// CHECK: #pragma omp declare mapper (id : struct foodat v) map(tofrom: v.a)
// CHECK: #pragma omp declare mapper (idd : struct foodatchild v) map(tofrom: v.k)
// CHECK: #pragma omp declare mapper (id : N1::vec v) map(tofrom: v.len)
// CHECK: {
// CHECK: #pragma omp declare mapper (id : N1::vec v) map(tofrom: v.len)
// CHECK: }
// CHECK: #pragma omp target map(mapper(id),alloc: fd)
// CHECK: #pragma omp target map(mapper(idd),alloc: fd.b)
// CHECK: #pragma omp target update to(mapper(id): fd)
// CHECK: #pragma omp target update to(mapper(idd): fd.b)
// CHECK: #pragma omp target update from(mapper(id): fd)
// CHECK: #pragma omp target update from(mapper(idd): fd.b)
// CHECK: }
// CHECK: template<> int foo<int>(int a) {
// CHECK: #pragma omp declare mapper (id : struct foodat v) map(tofrom: v.a)
// CHECK: #pragma omp declare mapper (idd : struct foodatchild v) map(tofrom: v.k)
// CHECK: #pragma omp declare mapper (id : N1::vec v) map(tofrom: v.len)
// CHECK: {
// CHECK: #pragma omp declare mapper (id : N1::vec v) map(tofrom: v.len)
// CHECK: }
// CHECK: #pragma omp target map(mapper(id),alloc: fd)
// CHECK: #pragma omp target map(mapper(idd),alloc: fd.b)
// CHECK: #pragma omp target update to(mapper(id): fd)
// CHECK: #pragma omp target update to(mapper(idd): fd.b)
// CHECK: #pragma omp target update from(mapper(id): fd)
// CHECK: #pragma omp target update from(mapper(idd): fd.b)
// CHECK: }

// CHECK: int main() {
int main() {
  N1::vec vv, vvv;
  N1::vecchild vc;
  dat<double> dd;
#pragma omp target map(mapper(N1::id) tofrom: vv) map(mapper(dat<double>::id) alloc: vvv)
// CHECK: #pragma omp target map(mapper(N1::id),tofrom: vv) map(mapper(dat<double>::id),alloc: vvv)
  { vv.len++; }
#pragma omp target map(mapper(N1::id) tofrom: vc)
// CHECK: #pragma omp target map(mapper(N1::id),tofrom: vc)
  { vc.len++; }
#pragma omp target map(mapper(default) tofrom: dd)
// CHECK: #pragma omp target map(mapper(default),tofrom: dd)
  { dd.d++; }

#pragma omp target update to(mapper(N1::id) : vc)
// CHECK: #pragma omp target update to(mapper(N1::id): vc)
#pragma omp target update to(mapper(dat<double>::id): vvv)
// CHECK: #pragma omp target update to(mapper(dat<double>::id): vvv)

#pragma omp target update from(mapper(N1::id) : vc)
// CHECK: #pragma omp target update from(mapper(N1::id): vc)
#pragma omp target update from(mapper(dat<double>::id): vvv)
// CHECK: #pragma omp target update from(mapper(dat<double>::id): vvv)

#pragma omp declare mapper(id: N1::vec v) map(v.len)
// CHECK: #pragma omp declare mapper (id : N1::vec v) map(tofrom: v.len)
  {
#pragma omp declare mapper(id: N1::vec v) map(v.len)
// CHECK: #pragma omp declare mapper (id : N1::vec v) map(tofrom: v.len)
  }
  return foo<int>(0);
}
// CHECK: }

#endif
