// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -ast-print %s -Wno-openmp-target | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s -Wno-openmp-target
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print -Wno-openmp-target | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -ast-print %s -Wno-openmp-target | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s -Wno-openmp-target
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print -Wno-openmp-target | FileCheck %s
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
#pragma omp target parallel for simd private(a) private(this->a) private(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
#pragma omp target parallel for simd private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp target parallel for simd private(this->a) private(this->a) private(T::a){{$}}
// CHECK: #pragma omp target parallel for simd private(this->a) private(this->a)
// CHECK: #pragma omp target parallel for simd private(this->a) private(this->a) private(this->S::a)

class S8 : public S7<S> {
  S8() {}

public:
  S8(int v) : S7<S>(v){
#pragma omp target parallel for simd private(a) private(this->a) private(S7<S>::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S8 &operator=(S8 &s) {
#pragma omp target parallel for simd private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp target parallel for simd private(this->a) private(this->a) private(this->S7<S>::a)
// CHECK: #pragma omp target parallel for simd private(this->a) private(this->a)

template <class T, int N>
T tmain(T argc, T *argv) {
  T b = argc, c, d, e, f, h;
  T arr[N][10], arr1[N];
  T i, j;
  T s;
  static T a;
// CHECK: static T a;
  static T g;
  const T clen = 5;
// CHECK: T clen = 5;
#pragma omp threadprivate(g)
#pragma omp target parallel for simd schedule(dynamic) default(none) linear(a) allocate(a)
  // CHECK: #pragma omp target parallel for simd schedule(dynamic) default(none) linear(a) allocate(a)
  for (T i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (T i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp target parallel for simd allocate(d) private(argc, b), firstprivate(c, d), lastprivate(d, f) collapse(N) schedule(static, N) ordered if (parallel :argc) num_threads(N) default(shared) shared(e) reduction(+ : h)
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int j = 0; j < 2; ++j)
        for (int j = 0; j < 2; ++j)
          for (int j = 0; j < 2; ++j)
            foo();
  // CHECK-NEXT: #pragma omp target parallel for simd allocate(d) private(argc,b) firstprivate(c,d) lastprivate(d,f) collapse(N) schedule(static, N) ordered if(parallel: argc) num_threads(N) default(shared) shared(e) reduction(+: h)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: foo();

#pragma omp target parallel for simd default(none), private(argc,b) firstprivate(argv) shared (d) if(parallel:argc > 0) num_threads(N) proc_bind(master) reduction(+:c, arr1[argc]) reduction(max:e, arr[:N][0:10])
  for (T i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(N) proc_bind(master) reduction(+: c,arr1[argc]) reduction(max: e,arr[:N][0:10])
  // CHECK-NEXT: for (T i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd if(N) num_threads(s) proc_bind(close) reduction(^:e, f, arr[0:N][:argc]) reduction(&& : h)
  for (T i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd if(N) num_threads(s) proc_bind(close) reduction(^: e,f,arr[0:N][:argc]) reduction(&&: h)
  // CHECK-NEXT: for (T i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd if(target:argc > 0)
  for (T i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd if(target: argc > 0)
  // CHECK-NEXT: for (T i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd if(parallel:argc > 0)
  for (T i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd if(parallel: argc > 0)
  // CHECK-NEXT: for (T i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd if(N)
  for (T i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd if(N)
  // CHECK-NEXT: for (T i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd map(i)
  for (T i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd map(tofrom: i)
  // CHECK-NEXT: for (T i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd map(arr1[0:10], i)
  for (T i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd map(tofrom: arr1[0:10],i)
  // CHECK-NEXT: for (T i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd map(to: i) map(from: j)
  for (T i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd map(to: i) map(from: j)
  // CHECK-NEXT: for (T i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd map(always,alloc: i)
  for (T i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd map(always,alloc: i)
  // CHECK-NEXT: for (T i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd nowait
  for (T i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd nowait
  // CHECK-NEXT: for (T i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd depend(in : argc, arr[i:argc], arr1[:])
  for (T i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd depend(in : argc,arr[i:argc],arr1[:])
  // CHECK-NEXT: for (T i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd defaultmap(tofrom: scalar)
  for (T i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd defaultmap(tofrom: scalar)
  // CHECK-NEXT: for (T i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd safelen(clen-1)
  for (T i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd safelen(clen - 1)
  // CHECK-NEXT: for (T i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd simdlen(clen-1)
  for (T i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd simdlen(clen - 1)
  // CHECK-NEXT: for (T i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd aligned(arr1:N-1)
  for (T i = 0; i < N; ++i) {}
  // CHECK: #pragma omp target parallel for simd aligned(arr1: N - 1)
  // CHECK-NEXT: for (T i = 0; i < N; ++i) {
  // CHECK-NEXT: }

  return T();
}

int main(int argc, char **argv) {
  int b = argc, c, d, e, f, h;
  int arr[5][10], arr1[5];
  int i, j;
  int s;
  static int a;
// CHECK: static int a;
  const int clen = 5;
// CHECK: int clen = 5;
  static float g;
#pragma omp threadprivate(g)
#pragma omp target parallel for simd schedule(guided, argc) default(none) linear(a)
  // CHECK: #pragma omp target parallel for simd schedule(guided, argc) default(none) linear(a)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;

#pragma omp target parallel for simd private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(2) schedule(auto) ordered if (target: argc) num_threads(a) default(shared) shared(e) reduction(+ : h) linear(a:-5)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      foo();
  // CHECK: #pragma omp target parallel for simd private(argc,b) firstprivate(argv,c) lastprivate(d,f) collapse(2) schedule(auto) ordered if(target: argc) num_threads(a) default(shared) shared(e) reduction(+: h) linear(a: -5)
  // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: for (int j = 0; j < 10; ++j)
  // CHECK-NEXT: foo();

#pragma omp target parallel for simd default(none), private(argc,b) firstprivate(argv) shared (d) if (parallel:argc > 0) num_threads(5) proc_bind(master) reduction(+:c, arr1[argc]) reduction(max:e, arr[:5][0:10])
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(5) proc_bind(master) reduction(+: c,arr1[argc]) reduction(max: e,arr[:5][0:10])
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd if (5) num_threads(s) proc_bind(close) reduction(^:e, f, arr[0:5][:argc]) reduction(&& : h)
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd if(5) num_threads(s) proc_bind(close) reduction(^: e,f,arr[0:5][:argc]) reduction(&&: h)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd if (target:argc > 0)
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd if(target: argc > 0)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd if (parallel:argc > 0)
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd if(parallel: argc > 0)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd if (5)
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd if(5)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd map(i)
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd  map(tofrom: i)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd map(arr1[0:10], i)
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd map(tofrom: arr1[0:10],i)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd map(to: i) map(from: j)
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd map(to: i) map(from: j)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd map(always,alloc: i)
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd map(always,alloc: i)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd nowait
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd nowait
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd depend(in : argc, arr[i:argc], arr1[:])
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd depend(in : argc,arr[i:argc],arr1[:])
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd defaultmap(tofrom: scalar)
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd defaultmap(tofrom: scalar)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd safelen(clen-1)
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd safelen(clen - 1)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd simdlen(clen-1)
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd simdlen(clen - 1)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

#pragma omp target parallel for simd aligned(arr1:4)
  for (int i = 0; i < 2; ++i) {}
  // CHECK: #pragma omp target parallel for simd aligned(arr1: 4)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }

  return (tmain<int, 5>(argc, &argc));
}

#endif
