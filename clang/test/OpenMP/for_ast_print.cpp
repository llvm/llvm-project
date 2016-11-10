// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
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
  T &b;
  typename T::type c:12;
  typename T::type &d;
  S7() : a(0), b(a), c(0), d(a.a) {}

public:
  S7(typename T::type v) : a(v), b(a), c(v), d(a.a) {
#pragma omp for private(a) private(this->a) private(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
#pragma omp for lastprivate(a) lastprivate(this->a) lastprivate(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
#pragma omp for linear(val(c))
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
#pragma omp for private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
#pragma omp for lastprivate(a) lastprivate(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
#pragma omp for linear(uval(this->b))
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp for private(this->a) private(this->a) private(T::a)
// CHECK: #pragma omp for lastprivate(this->a) lastprivate(this->a) lastprivate(T::a)
// CHECK: #pragma omp for linear(val(this->c))
// CHECK: #pragma omp for private(this->a) private(this->a)
// CHECK: #pragma omp for lastprivate(this->a) lastprivate(this->a)
// CHECK: #pragma omp for linear(uval(this->b))
// CHECK: #pragma omp for private(this->a) private(this->a) private(this->S::a)
// CHECK: #pragma omp for lastprivate(this->a) lastprivate(this->a) lastprivate(this->S::a)
// CHECK: #pragma omp for linear(val(this->c))

class S8 : public S7<S> {
  S8() {}

public:
  S8(int v) : S7<S>(v){
#pragma omp for private(a) private(this->a) private(S7<S>::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
#pragma omp for lastprivate(a) lastprivate(this->a) lastprivate(S7<S>::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
#pragma omp for linear(ref(S7<S>::d))
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S8 &operator=(S8 &s) {
#pragma omp for private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
#pragma omp for lastprivate(a) lastprivate(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
#pragma omp for linear(this->c)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp for private(this->a) private(this->a) private(this->S7<S>::a)
// CHECK: #pragma omp for lastprivate(this->a) lastprivate(this->a) lastprivate(this->S7<S>::a)
// CHECK: #pragma omp for linear(ref(this->S7<S>::d))
// CHECK: #pragma omp for private(this->a) private(this->a)
// CHECK: #pragma omp for lastprivate(this->a) lastprivate(this->a)
// CHECK: #pragma omp for linear(this->c)

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, g;
  static T a;
// CHECK: static T a;
#pragma omp for schedule(dynamic) linear(a)
  // CHECK-NEXT: #pragma omp for schedule(dynamic) linear(a)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp parallel
#pragma omp for private(argc, b), firstprivate(c, d), lastprivate(d, f) collapse(N) schedule(static, N) ordered(N) nowait
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
  // CHECK-NEXT: #pragma omp parallel
  // CHECK-NEXT: #pragma omp for private(argc,b) firstprivate(c,d) lastprivate(d,f) collapse(N) schedule(static, N) ordered(N) nowait
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
  return T();
}

int main(int argc, char **argv) {
// CHECK: int main(int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  static int a;
// CHECK: static int a;
#pragma omp for schedule(guided, argc)
  // CHECK-NEXT: #pragma omp for schedule(guided, argc)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp parallel
#pragma omp for private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(2) schedule(auto) ordered nowait linear(g:-1)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      foo();
  // CHECK-NEXT: #pragma omp parallel
  // CHECK-NEXT: #pragma omp for private(argc,b) firstprivate(argv,c) lastprivate(d,f) collapse(2) schedule(auto) ordered nowait linear(g: -1)
  // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: for (int j = 0; j < 10; ++j)
  // CHECK-NEXT: foo();
  char buf[9] = "01234567";
  char *p, *q;
#pragma omp parallel
#pragma omp for
  for (p = buf; p < &buf[8]; p++)
    for (q = &buf[0]; q <= buf + 7; q++)
      foo();
  // CHECK: #pragma omp parallel
  // CHECK-NEXT: #pragma omp for
  // CHECK-NEXT: for (p = buf; p < &buf[8]; p++)
  // CHECK-NEXT: for (q = &buf[0]; q <= buf + 7; q++)
  // CHECK-NEXT: foo();
  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
