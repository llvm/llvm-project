// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

struct S1 {
  S1(): a(0) {}
  S1(int v) : a(v) {}
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
#pragma omp task private(a) private(this->a) private(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
#pragma omp task private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp task private(this->a) private(this->a) private(this->S1::a)
// CHECK: #pragma omp task private(this->a) private(this->a) private(T::a)
// CHECK: #pragma omp task private(this->a) private(this->a)

class S8 : public S7<S1> {
  S8() {}

public:
  S8(int v) : S7<S1>(v){
#pragma omp task private(a) private(this->a) private(S7<S1>::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S8 &operator=(S8 &s) {
#pragma omp task private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp task private(this->a) private(this->a) private(this->S7<S1>::a)
// CHECK: #pragma omp task private(this->a) private(this->a)

template <class T>
struct S {
  operator T() { return T(); }
  static T TS;
#pragma omp threadprivate(TS)
};

// CHECK:      template <class T = int> struct S {
// CHECK:        static int TS;
// CHECK-NEXT:   #pragma omp threadprivate(S<int>::TS)
// CHECK-NEXT: }
// CHECK:      template <class T = long> struct S {
// CHECK:        static long TS;
// CHECK-NEXT:   #pragma omp threadprivate(S<long>::TS)
// CHECK-NEXT: }
// CHECK:      template <class T> struct S {
// CHECK:        static T TS;
// CHECK-NEXT:   #pragma omp threadprivate(S::TS)
// CHECK:      };

template <typename T, int C>
T tmain(T argc, T *argv) {
  T b = argc, c, d, e, f, g;
  static T a;
  S<T> s;
  T arr[argc];
#pragma omp task untied depend(in : argc, argv[b:argc], arr[:]) if (task : argc > 0)
  a = 2;
#pragma omp task default(none), private(argc, b) firstprivate(argv) shared(d) if (argc > 0) final(S<T>::TS > 0) priority(argc)
  foo();
#pragma omp task if (C) mergeable priority(C)
  foo();
  return 0;
}

// CHECK: template <typename T = int, int C = 5> int tmain(int argc, int *argv) {
// CHECK-NEXT: int b = argc, c, d, e, f, g;
// CHECK-NEXT: static int a;
// CHECK-NEXT: S<int> s;
// CHECK-NEXT: int arr[argc];
// CHECK-NEXT: #pragma omp task untied depend(in : argc,argv[b:argc],arr[:]) if(task: argc > 0)
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp task default(none) private(argc,b) firstprivate(argv) shared(d) if(argc > 0) final(S<int>::TS > 0) priority(argc)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp task if(5) mergeable priority(5)
// CHECK-NEXT: foo()
// CHECK: template <typename T = long, int C = 1> long tmain(long argc, long *argv) {
// CHECK-NEXT: long b = argc, c, d, e, f, g;
// CHECK-NEXT: static long a;
// CHECK-NEXT: S<long> s;
// CHECK-NEXT: long arr[argc];
// CHECK-NEXT: #pragma omp task untied depend(in : argc,argv[b:argc],arr[:]) if(task: argc > 0)
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp task default(none) private(argc,b) firstprivate(argv) shared(d) if(argc > 0) final(S<long>::TS > 0) priority(argc)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp task if(1) mergeable priority(1)
// CHECK-NEXT: foo()
// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T b = argc, c, d, e, f, g;
// CHECK-NEXT: static T a;
// CHECK-NEXT: S<T> s;
// CHECK-NEXT: T arr[argc];
// CHECK-NEXT: #pragma omp task untied depend(in : argc,argv[b:argc],arr[:]) if(task: argc > 0)
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp task default(none) private(argc,b) firstprivate(argv) shared(d) if(argc > 0) final(S<T>::TS > 0) priority(argc)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp task if(C) mergeable priority(C)
// CHECK-NEXT: foo()

enum Enum {};

int main(int argc, char **argv) {
  long x;
  int b = argc, c, d, e, f, g;
  static int a;
  int arr[10];
#pragma omp threadprivate(a)
  Enum ee;
// CHECK: Enum ee;
#pragma omp task untied mergeable depend(out:argv[:a][1], (arr)[0:]) if(task: argc > 0) priority(f)
  // CHECK-NEXT: #pragma omp task untied mergeable depend(out : argv[:a][1],(arr)[0:]) if(task: argc > 0) priority(f)
  a = 2;
// CHECK-NEXT: a = 2;
#pragma omp task default(none), private(argc, b) firstprivate(argv) if (argc > 0) final(a > 0) depend(inout : a, argv[:argc],arr[:a]) priority(23)
  // CHECK-NEXT: #pragma omp task default(none) private(argc,b) firstprivate(argv) if(argc > 0) final(a > 0) depend(inout : a,argv[:argc],arr[:a]) priority(23)
  foo();
  // CHECK-NEXT: foo();
  return tmain<int, 5>(b, &b) + tmain<long, 1>(x, &x);
}

extern template int S<int>::TS;
extern template long S<long>::TS;

#endif
