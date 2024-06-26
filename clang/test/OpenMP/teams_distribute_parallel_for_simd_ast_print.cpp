// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -ast-print %s -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix OMP45
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -include-pch %t -verify %s -ast-print -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix OMP45
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -DOMP5 -ast-print %s -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix OMP50
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -DOMP5 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -DOMP5 -std=c++11 -include-pch %t -verify %s -ast-print -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix OMP50
// RUN: %clang_cc1 -verify -fopenmp -DOMP51 -ast-print %s -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix OMP51
// RUN: %clang_cc1 -fopenmp -DOMP51 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -DOMP51 -std=c++11 -include-pch %t -verify %s -ast-print -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix OMP51

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -ast-print %s -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix OMP45
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -std=c++11 -include-pch %t -verify %s -ast-print -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix OMP45
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -DOMP5 -ast-print %s -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix OMP50
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -DOMP5 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -DOMP5 -std=c++11 -include-pch %t -verify %s -ast-print -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix OMP50
// RUN: %clang_cc1 -verify -fopenmp-simd -DOMP51 -ast-print %s -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix OMP51
// RUN: %clang_cc1 -fopenmp-simd -DOMP51 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -DOMP51 -std=c++11 -include-pch %t -verify %s -ast-print -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix OMP51
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

struct S {
  S(): a(0) {}
  S(int v) : a(v) {}
  int a;
  typedef int type;
};

template <typename T>
class S7 : public T {
protected:
  T a;
  S7() : a(0) {}

public:
  S7(typename T::type v) : a(v) {
#pragma omp target
#pragma omp teams distribute parallel for simd private(a) private(this->a) private(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
    int k;
#pragma omp target
#pragma omp teams distribute parallel for simd private(a) private(this->a) linear(k)
    for (k = 0; k < s.a.a; ++k)
      ++s.a.a;

    foo();
    bar();
    return *this;
  }
  void foo() {
    int b, argv, d, c, e, f;
#pragma omp target
#pragma omp teams distribute parallel for simd default(none), private(b) firstprivate(argv) shared(d) reduction(+:c) reduction(max:e) num_teams(f) thread_limit(d)
    for (int k = 0; k < a.a; ++k)
      ++a.a;
  }
  void bar() {
    int arr[10];
    const int alen = 16;
    const int slen1 = 8;
    const int slen2 = 8;
#pragma omp target
#pragma omp teams distribute parallel for simd simdlen(slen1) safelen(slen2) aligned(arr:alen)
    for (int k = 0; k < a.a; ++k)
      ++a.a;
  }
};
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd private(this->a) private(this->a) private(T::a)
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd private(this->a) private(this->a) linear(k)
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd default(none) private(b) firstprivate(argv) shared(d) reduction(+: c) reduction(max: e) num_teams(f) thread_limit(d)
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd simdlen(slen1) safelen(slen2) aligned(arr: alen)

class S8 : public S7<S> {
  S8() {}

public:
  S8(int v) : S7<S>(v){
#pragma omp target
#pragma omp teams distribute parallel for simd private(a) private(this->a) private(S7<S>::a) 
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S8 &operator=(S8 &s) {
#pragma omp target
#pragma omp teams distribute parallel for simd private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;

    foo();
    bar();
    return *this;
  }
  void bar() {
    int b, argv, d, c, e, f8;
#pragma omp target
#pragma omp teams distribute parallel for simd default(none), private(b) firstprivate(argv) shared(d) reduction(+:c) reduction(max:e) num_teams(f8) thread_limit(d)
    for (int k = 0; k < a.a; ++k)
      ++a.a;
  }
  void foo() {
    const int alen = 16;
    const int slen1 = 8;
    const int slen2 = 8;
    int arr[10];
#pragma omp target
#pragma omp teams distribute parallel for simd simdlen(slen1) safelen(slen2) aligned(arr:alen)
    for (int k = 0; k < a.a; ++k)
      ++a.a;
  }
};
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd private(this->a) private(this->a) private(this->S::a)
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd private(this->a) private(this->a) private(this->S7<S>::a)
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd private(this->a) private(this->a)
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd default(none) private(b) firstprivate(argv) shared(d) reduction(+: c) reduction(max: e) num_teams(f8) thread_limit(d)
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd simdlen(slen1) safelen(slen2) aligned(arr: alen)

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, g;
  static T a;
// CHECK: static T a;
  const T clen = 5;
  const T alen = 16;
  int arr[10];
#pragma omp target
#pragma omp teams distribute parallel for simd
  for (int i=0; i < 2; ++i)
    a = 2;
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd{{$}}
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp target
#pragma omp teams distribute parallel for simd allocate(b) private(argc, b), firstprivate(c, d), collapse(2) allocate(d)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      foo();
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd allocate(b) private(argc,b) firstprivate(c,d) collapse(2) allocate(d)
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: for (int j = 0; j < 10; ++j)
// CHECK-NEXT: foo();
  for (int i = 0; i < 10; ++i)
    foo();
// CHECK: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();
#pragma omp target
#ifdef OMP51
#pragma omp teams distribute parallel for simd if(simd:argc) nontemporal(argc, c, d) order(reproducible:concurrent)
#elif OMP5
#pragma omp teams distribute parallel for simd if(simd:argc) nontemporal(argc, c, d) order(concurrent)
#else
#pragma omp teams distribute parallel for simd
#endif // OMP51
  for (int i = 0; i < 10; ++i)
    foo();
// CHECK: #pragma omp target
// OMP45-NEXT: #pragma omp teams distribute parallel for simd
// OMP50-NEXT: #pragma omp teams distribute parallel for simd if(simd: argc) nontemporal(argc,c,d) order(concurrent)
// OMP51-NEXT: #pragma omp teams distribute parallel for simd if(simd: argc) nontemporal(argc,c,d) order(reproducible: concurrent)
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();
#pragma omp target
#pragma omp teams distribute parallel for simd default(none), private(b) firstprivate(argc) shared(d) reduction(+:c) reduction(max:e) num_teams(f) thread_limit(d)
    for (int k = 0; k < 10; ++k)
      e += d + argc;
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd default(none) private(b) firstprivate(argc) shared(d) reduction(+: c) reduction(max: e) num_teams(f) thread_limit(d)
// CHECK-NEXT: for (int k = 0; k < 10; ++k)
// CHECK-NEXT: e += d + argc;
#pragma omp target
#pragma omp teams distribute parallel for simd simdlen(clen-1)
  for (int k = 0; k < 10; ++k)
    e += d + argc;
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd simdlen(clen - 1)
// CHECK-NEXT: for (int k = 0; k < 10; ++k)
// CHECK-NEXT: e += d + argc;
#pragma omp target
#pragma omp teams distribute parallel for simd safelen(clen-1) aligned(arr:alen)
  for (int k = 0; k < 10; ++k)
    e += d + argc + arr[k];
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd safelen(clen - 1) aligned(arr: alen)
// CHECK-NEXT: for (int k = 0; k < 10; ++k)
// CHECK-NEXT: e += d + argc + arr[k];
  return T();
}

int main (int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  static int a;
// CHECK: static int a;
  const int clen = 5;
  const int N = 10;
  int arr[10];
#pragma omp target
#pragma omp teams distribute parallel for simd
  for (int i=0; i < 2; ++i)
    a = 2;
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp target
#pragma omp teams distribute parallel for simd private(argc,b),firstprivate(argv, c), collapse(2)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      foo();
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd private(argc,b) firstprivate(argv,c) collapse(2)
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: for (int j = 0; j < 10; ++j)
// CHECK-NEXT: foo();
  for (int i = 0; i < 10; ++i)
    foo();
// CHECK: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();
#pragma omp target
#pragma omp teams distribute parallel for simd
  for (int i = 0; i < 10; ++i)foo();
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();
#pragma omp target
#pragma omp teams distribute parallel for simd default(none), private(b) firstprivate(argc) shared(d) reduction(+:c) reduction(max:e) num_teams(f) thread_limit(d)
  for (int k = 0; k < 10; ++k)
    e += d + argc;
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd default(none) private(b) firstprivate(argc) shared(d) reduction(+: c) reduction(max: e) num_teams(f) thread_limit(d)
// CHECK-NEXT: for (int k = 0; k < 10; ++k)
// CHECK-NEXT: e += d + argc;
#pragma omp target
#pragma omp teams distribute parallel for simd simdlen(clen-1)
  for (int k = 0; k < 10; ++k)
    e += d + argc;
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd simdlen(clen - 1)
// CHECK-NEXT: for (int k = 0; k < 10; ++k)
// CHECK-NEXT: e += d + argc;
#pragma omp target
#pragma omp teams distribute parallel for simd safelen(clen-1) aligned(arr:N+6)
  for (int k = 0; k < 10; ++k)
    e += d + argc + arr[k];
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for simd safelen(clen - 1) aligned(arr: N + 6)
// CHECK-NEXT: for (int k = 0; k < 10; ++k)
// CHECK-NEXT: e += d + argc + arr[k];
  return (0);
}

#endif
