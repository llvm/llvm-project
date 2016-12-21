// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
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
#pragma omp target parallel for private(a) private(this->a) private(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
#pragma omp target parallel for private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp target parallel for private(this->a) private(this->a) private(T::a)
// CHECK: #pragma omp target parallel for private(this->a) private(this->a)
// CHECK: #pragma omp target parallel for private(this->a) private(this->a) private(this->S::a)

class S8 : public S7<S> {
  S8() {}

public:
  S8(int v) : S7<S>(v){
#pragma omp target parallel for private(a) private(this->a) private(S7<S>::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S8 &operator=(S8 &s) {
#pragma omp target parallel for private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp target parallel for private(this->a) private(this->a) private(this->S7<S>::a)
// CHECK: #pragma omp target parallel for private(this->a) private(this->a)

template <class T, int N>
T tmain(T argc, T *argv) {
  T b = argc, c, d, e, f, h;
  T arr[N][10], arr1[N];
  T i, j;
  T s;
  static T a;
// CHECK: static T a;
  static T g;
#pragma omp threadprivate(g)
#pragma omp target parallel for schedule(dynamic) default(none) linear(a)
  // CHECK: #pragma omp target parallel for schedule(dynamic) default(none) linear(a)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp target parallel for private(argc, b), firstprivate(c, d), lastprivate(d, f) collapse(N) schedule(static, N) ordered(N) if (parallel :argc) num_threads(N) default(shared) shared(e) reduction(+ : h)
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int j = 0; j < 2; ++j)
        for (int j = 0; j < 2; ++j)
          for (int j = 0; j < 2; ++j)
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int j = 0; j < 2; ++j)
        for (int j = 0; j < 2; ++j)
          for (int j = 0; j < 2; ++j)
            foo();
  // CHECK-NEXT: #pragma omp target parallel for private(argc,b) firstprivate(c,d) lastprivate(d,f) collapse(N) schedule(static, N) ordered(N) if(parallel: argc) num_threads(N) default(shared) shared(e) reduction(+: h)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: foo();
#pragma omp target parallel for default(none), private(argc,b) firstprivate(argv) shared (d) if (parallel:argc > 0) num_threads(N) proc_bind(master) reduction(+:c, arr1[argc]) reduction(max:e, arr[:N][0:10])
  for (int i = 0; i < 2; ++i) {}
// CHECK-NEXT: #pragma omp target parallel for default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(N) proc_bind(master) reduction(+: c,arr1[argc]) reduction(max: e,arr[:N][0:10])
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for if (N) num_threads(s) proc_bind(close) reduction(^:e, f, arr[0:N][:argc]) reduction(&& : h)
// CHECK-NEXT: #pragma omp target parallel for if(N) num_threads(s) proc_bind(close) reduction(^: e,f,arr[0:N][:argc]) reduction(&&: h)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for if (target:argc > 0)
// CHECK-NEXT: #pragma omp target parallel for if(target: argc > 0)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for if (parallel:argc > 0)
// CHECK-NEXT: #pragma omp target parallel for if(parallel: argc > 0)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for if (N)
// CHECK-NEXT: #pragma omp target parallel for if(N)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for map(i)
// CHECK-NEXT: #pragma omp target parallel for map(tofrom: i)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for map(arr1[0:10], i)
// CHECK-NEXT: #pragma omp target parallel for map(tofrom: arr1[0:10],i)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for map(to: i) map(from: j)
// CHECK-NEXT: #pragma omp target parallel for map(to: i) map(from: j)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for map(always,alloc: i)
// CHECK-NEXT: #pragma omp target parallel for map(always,alloc: i)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for nowait
// CHECK-NEXT: #pragma omp target parallel for nowait
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for depend(in : argc, arr[i:argc], arr1[:])
// CHECK-NEXT: #pragma omp target parallel for depend(in : argc,arr[i:argc],arr1[:])
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for defaultmap(tofrom: scalar)
// CHECK-NEXT: #pragma omp target parallel for defaultmap(tofrom: scalar)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
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
  static float g;
#pragma omp threadprivate(g)
#pragma omp target parallel for schedule(guided, argc) default(none) linear(a)
  // CHECK: #pragma omp target parallel for schedule(guided, argc) default(none) linear(a)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp target parallel for private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(2) schedule(auto) ordered if (target: argc) num_threads(a) default(shared) shared(e) reduction(+ : h) linear(a:-5)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      foo();
  // CHECK-NEXT: #pragma omp target parallel for private(argc,b) firstprivate(argv,c) lastprivate(d,f) collapse(2) schedule(auto) ordered if(target: argc) num_threads(a) default(shared) shared(e) reduction(+: h) linear(a: -5)
 // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: for (int j = 0; j < 10; ++j)
  // CHECK-NEXT: foo();
#pragma omp target parallel for default(none), private(argc,b) firstprivate(argv) shared (d) if (parallel:argc > 0) num_threads(5) proc_bind(master) reduction(+:c, arr1[argc]) reduction(max:e, arr[:5][0:10])
  for (int i = 0; i < 2; ++i) {}
// CHECK-NEXT: #pragma omp target parallel for default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(5) proc_bind(master) reduction(+: c,arr1[argc]) reduction(max: e,arr[:5][0:10])
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for if (5) num_threads(s) proc_bind(close) reduction(^:e, f, arr[0:5][:argc]) reduction(&& : h)
// CHECK-NEXT: #pragma omp target parallel for if(5) num_threads(s) proc_bind(close) reduction(^: e,f,arr[0:5][:argc]) reduction(&&: h)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for if (target:argc > 0)
// CHECK-NEXT: #pragma omp target parallel for if(target: argc > 0)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for if (parallel:argc > 0)
// CHECK-NEXT: #pragma omp target parallel for if(parallel: argc > 0)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for if (5)
// CHECK-NEXT: #pragma omp target parallel for if(5)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for map(i)
// CHECK-NEXT: #pragma omp target parallel for map(tofrom: i)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for map(arr1[0:10], i)
// CHECK-NEXT: #pragma omp target parallel for map(tofrom: arr1[0:10],i)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for map(to: i) map(from: j)
// CHECK-NEXT: #pragma omp target parallel for map(to: i) map(from: j)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for map(always,alloc: i)
// CHECK-NEXT: #pragma omp target parallel for map(always,alloc: i)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for nowait
// CHECK-NEXT: #pragma omp target parallel for nowait
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for depend(in : argc, arr[i:argc], arr1[:])
// CHECK-NEXT: #pragma omp target parallel for depend(in : argc,arr[i:argc],arr1[:])
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
#pragma omp target parallel for defaultmap(tofrom: scalar)
// CHECK-NEXT: #pragma omp target parallel for defaultmap(tofrom: scalar)
  for (int i = 0; i < 2; ++i) {}
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: }
  return (tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]));
}

#endif
