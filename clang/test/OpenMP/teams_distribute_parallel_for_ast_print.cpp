// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

int x;
#pragma omp threadprivate(x)

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
#pragma omp teams distribute parallel for private(a) private(this->a) private(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
#pragma omp target
#pragma omp teams distribute parallel for private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;

    foo();
    return *this;
  }
  void foo() {
    int b, argv, d, c, e, f;
#pragma omp target
#pragma omp teams distribute parallel for default(none), private(b) firstprivate(argv) shared(d) reduction(+:c) reduction(max:e) num_teams(f) thread_limit(d) copyin(x)
    for (int k = 0; k < a.a; ++k)
      ++a.a;
  }
};
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for private(this->a) private(this->a) private(T::a)
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for private(this->a) private(this->a)
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for default(none) private(b) firstprivate(argv) shared(d) reduction(+: c) reduction(max: e) num_teams(f) thread_limit(d) copyin(x)

class S8 : public S7<S> {
  S8() {}

public:
  S8(int v) : S7<S>(v){
#pragma omp target
#pragma omp teams distribute parallel for private(a) private(this->a) private(S7<S>::a) 
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S8 &operator=(S8 &s) {
#pragma omp target
#pragma omp teams distribute parallel for private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;

    bar();
    return *this;
  }
  void bar() {
    int b, argv, d, c, e, f8;
#pragma omp target
#pragma omp teams distribute parallel for default(none), private(b) firstprivate(argv) shared(d) reduction(+:c) reduction(max:e) num_teams(f8) thread_limit(d) copyin(x)
    for (int k = 0; k < a.a; ++k)
      ++a.a;
  }
};
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for private(this->a) private(this->a) private(this->S::a)
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for private(this->a) private(this->a) private(this->S7<S>::a)
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for private(this->a) private(this->a)
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for default(none) private(b) firstprivate(argv) shared(d) reduction(+: c) reduction(max: e) num_teams(f8) thread_limit(d) copyin(x)

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, g;
  static T a;
// CHECK: static T a;
  const T clen = 5;
  const T alen = 16;
  int arr[10];
#pragma omp target
#pragma omp teams distribute parallel for
  for (int i=0; i < 2; ++i)
    a = 2;
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp target
#pragma omp teams distribute parallel for private(argc, b), firstprivate(c, d), collapse(2)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      foo();
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for private(argc,b) firstprivate(c,d) collapse(2)
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: for (int j = 0; j < 10; ++j)
// CHECK-NEXT: foo();
  for (int i = 0; i < 10; ++i)
    foo();
// CHECK: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();
#pragma omp target
#pragma omp teams distribute parallel for
  for (int i = 0; i < 10; ++i)
    foo();
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();  
#pragma omp target
#pragma omp teams distribute parallel for default(none), private(b) firstprivate(argc) shared(d) reduction(+:c) reduction(max:e) num_teams(f) thread_limit(d) copyin(x)
    for (int k = 0; k < 10; ++k)
      e += d + argc;
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for default(none) private(b) firstprivate(argc) shared(d) reduction(+: c) reduction(max: e) num_teams(f) thread_limit(d) copyin(x)
// CHECK-NEXT: for (int k = 0; k < 10; ++k)
// CHECK-NEXT: e += d + argc;
#pragma omp target
#pragma omp teams distribute parallel for linear(d)
  for (int k = 0; k < 10; ++k)
    e += d + argc;
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for linear(d)
// CHECK-NEXT: for (int k = 0; k < 10; ++k)
// CHECK-NEXT: e += d + argc;
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
#pragma omp teams distribute parallel for
  for (int i=0; i < 2; ++i)
    a = 2;
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp target
#pragma omp teams distribute parallel for private(argc,b),firstprivate(argv, c), collapse(2)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      foo();
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for private(argc,b) firstprivate(argv,c) collapse(2)
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: for (int j = 0; j < 10; ++j)
// CHECK-NEXT: foo();
  for (int i = 0; i < 10; ++i)
    foo();
// CHECK: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();
#pragma omp target
#pragma omp teams distribute parallel for
  for (int i = 0; i < 10; ++i)foo();
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();
#pragma omp target
#pragma omp teams distribute parallel for default(none), private(b) firstprivate(argc) shared(d) reduction(+:c) reduction(max:e) num_teams(f) thread_limit(d) copyin(x)
  for (int k = 0; k < 10; ++k)
    e += d + argc;
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for default(none) private(b) firstprivate(argc) shared(d) reduction(+: c) reduction(max: e) num_teams(f) thread_limit(d) copyin(x)
// CHECK-NEXT: for (int k = 0; k < 10; ++k)
// CHECK-NEXT: e += d + argc;
#pragma omp target
#pragma omp teams distribute parallel for linear(d)
  for (int k = 0; k < 10; ++k)
    e += d + argc;
// CHECK: #pragma omp target
// CHECK-NEXT: #pragma omp teams distribute parallel for linear(d)
// CHECK-NEXT: for (int k = 0; k < 10; ++k)
// CHECK-NEXT: e += d + argc;
  return (0);
}

#endif
