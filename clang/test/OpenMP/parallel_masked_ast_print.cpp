// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

struct S1 {
  S1(): a(0) {}
  S1(int v) : a(v) {}
  int a;
  typedef int type;
  S1& operator +(const S1&);
  S1& operator *(const S1&);
  S1& operator &&(const S1&);
  S1& operator ^(const S1&);
};

template <typename T>
class S7 : public T {
protected:
  T a;
  T b[100];
  S7() : a(0) {}

public:
  S7(typename T::type v) : a(v) {
    int tid = 0;
// CHECK: #pragma omp parallel masked filter(tid) private(this->a) private(this->a) private(T::a)
#pragma omp parallel masked filter(tid) private(a) private(this->a) private(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
// CHECK: #pragma omp parallel masked filter(tid) firstprivate(this->a) firstprivate(this->a) firstprivate(T::a)
#pragma omp parallel masked filter(tid) firstprivate(a) firstprivate(this->a) firstprivate(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
// CHECK: #pragma omp parallel masked filter(tid) shared(this->a) shared(this->a) shared(T::a)
#pragma omp parallel masked filter(tid) shared(a) shared(this->a) shared(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
// CHECK: #pragma omp parallel masked filter(tid) reduction(+: this->a) reduction(*: this->b[:])
#pragma omp parallel masked filter(tid) reduction(+ : a) reduction(*: b[:])
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
    int tid = 0;
// CHECK: #pragma omp parallel masked filter(tid) private(this->a) private(this->a)
#pragma omp parallel masked filter(tid) private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
// CHECK: #pragma omp parallel masked filter(tid) firstprivate(this->a) firstprivate(this->a)
#pragma omp parallel masked filter(tid) firstprivate(a) firstprivate(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
// CHECK: #pragma omp parallel masked filter(tid) shared(this->a) shared(this->a)
#pragma omp parallel masked filter(tid) shared(a) shared(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
// CHECK: #pragma omp parallel masked filter(tid) reduction(&&: this->a) reduction(^: this->b[s.a.a])
#pragma omp parallel masked filter(tid) reduction(&& : this->a) reduction(^: b[s.a.a])
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp parallel masked filter(tid) private(this->a) private(this->a) private(this->S1::a)
// CHECK: #pragma omp parallel masked filter(tid) firstprivate(this->a) firstprivate(this->a) firstprivate(this->S1::a)
// CHECK: #pragma omp parallel masked filter(tid) shared(this->a) shared(this->a) shared(this->S1::a)
// CHECK: #pragma omp parallel masked filter(tid) reduction(+: this->a) reduction(*: this->b[:])

class S8 : public S7<S1> {
  S8() {}

public:
  S8(int v) : S7<S1>(v){
// CHECK: #pragma omp parallel masked private(this->a) private(this->a) private(this->S7<S1>::a)
#pragma omp parallel masked private(a) private(this->a) private(S7 < S1 > ::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
// CHECK: #pragma omp parallel masked firstprivate(this->a) firstprivate(this->a) firstprivate(this->S7<S1>::a)
#pragma omp parallel masked firstprivate(a) firstprivate(this->a) firstprivate(S7 < S1 > ::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
// CHECK: #pragma omp parallel masked shared(this->a) shared(this->a) shared(this->S7<S1>::a)
#pragma omp parallel masked shared(a) shared(this->a) shared(S7 < S1 > ::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
// CHECK: #pragma omp parallel masked reduction(^: this->S7<S1>::a) reduction(+: this->S7<S1>::b[:this->S7<S1>::a.a])
#pragma omp parallel masked reduction(^ : S7 < S1 > ::a) reduction(+ : S7 < S1 > ::b[ : S7 < S1 > ::a.a])
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S8 &operator=(S8 &s) {
// CHECK: #pragma omp parallel masked private(this->a) private(this->a)
#pragma omp parallel masked private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
// CHECK: #pragma omp parallel masked firstprivate(this->a) firstprivate(this->a)
#pragma omp parallel masked firstprivate(a) firstprivate(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
// CHECK: #pragma omp parallel masked shared(this->a) shared(this->a)
#pragma omp parallel masked shared(a) shared(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
// CHECK: #pragma omp parallel masked reduction(*: this->a) reduction(&&: this->b[this->a.a:])
#pragma omp parallel masked reduction(* : this->a) reduction(&&:this->b[a.a:])
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};


template <class T>
struct S {
  operator T() {return T();}
  static T TS;
  #pragma omp threadprivate(TS)
};

// CHECK:      template <class T> struct S {
// CHECK:        static T TS;
// CHECK-NEXT:   #pragma omp threadprivate(S::TS)
// CHECK:      };
// CHECK:      template<> struct S<int> {
// CHECK:        static int TS;
// CHECK-NEXT:   #pragma omp threadprivate(S<int>::TS)
// CHECK-NEXT: }
// CHECK:      template<> struct S<long> {
// CHECK:        static long TS;
// CHECK-NEXT:   #pragma omp threadprivate(S<long>::TS)
// CHECK-NEXT: }

int thrp;
#pragma omp threadprivate(thrp)

template <typename T, int C>
T tmain(T argc, T *argv) {
  T b = argc, c, d, e, f, g;
  static T a;
  S<T> s;
  T arr[C][10], arr1[C];
#pragma omp parallel masked
  a=2;
#pragma omp parallel masked default(none), private(argc,b) firstprivate(argv) shared (d) if (parallel:argc > 0) num_threads(C) copyin(S<T>::TS, thrp) proc_bind(master) reduction(+:c, arr1[argc]) reduction(max:e, arr[:C][0:10])
  foo();
#pragma omp parallel masked if (C) num_threads(s) proc_bind(close) reduction(^:e, f, arr[0:C][:argc]) reduction(task, && : g)
  foo();
  return 0;
}

// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T b = argc, c, d, e, f, g;
// CHECK-NEXT: static T a;
// CHECK-NEXT: S<T> s;
// CHECK-NEXT: T arr[C][10], arr1[C];
// CHECK-NEXT: #pragma omp parallel masked
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp parallel masked default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(C) copyin(S<T>::TS,thrp) proc_bind(master) reduction(+: c,arr1[argc]) reduction(max: e,arr[:C][0:10])
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp parallel masked if(C) num_threads(s) proc_bind(close) reduction(^: e,f,arr[0:C][:argc]) reduction(task, &&: g)
// CHECK-NEXT: foo()
// CHECK: template<> int tmain<int, 5>(int argc, int *argv) {
// CHECK-NEXT: int b = argc, c, d, e, f, g;
// CHECK-NEXT: static int a;
// CHECK-NEXT: S<int> s;
// CHECK-NEXT: int arr[5][10], arr1[5];
// CHECK-NEXT: #pragma omp parallel masked
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp parallel masked default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(5) copyin(S<int>::TS,thrp) proc_bind(master) reduction(+: c,arr1[argc]) reduction(max: e,arr[:5][0:10])
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp parallel masked if(5) num_threads(s) proc_bind(close) reduction(^: e,f,arr[0:5][:argc]) reduction(task, &&: g)
// CHECK-NEXT: foo()
// CHECK: template<> long tmain<long, 1>(long argc, long *argv) {
// CHECK-NEXT: long b = argc, c, d, e, f, g;
// CHECK-NEXT: static long a;
// CHECK-NEXT: S<long> s;
// CHECK-NEXT: long arr[1][10], arr1[1];
// CHECK-NEXT: #pragma omp parallel masked
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp parallel masked default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(1) copyin(S<long>::TS,thrp) proc_bind(master) reduction(+: c,arr1[argc]) reduction(max: e,arr[:1][0:10])
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp parallel masked if(1) num_threads(s) proc_bind(close) reduction(^: e,f,arr[0:1][:argc]) reduction(task, &&: g)
// CHECK-NEXT: foo()

enum Enum { };

int main (int argc, char **argv) {
  long x;
  int b = argc, c, d, e, f, g;
  static int a;
  #pragma omp threadprivate(a)
  int arr[10][argc], arr1[2];
  Enum ee;
// CHECK: Enum ee;
#pragma omp parallel masked
// CHECK-NEXT: #pragma omp parallel masked
  a=2;
// CHECK-NEXT: a = 2;
#pragma omp parallel masked default(none), private(argc,b) firstprivate(argv) if (parallel: argc > 0) num_threads(ee) copyin(a) proc_bind(spread) reduction(| : c, d, arr1[argc]) reduction(* : e, arr[:10][0:argc]) allocate(e)
// CHECK-NEXT: #pragma omp parallel masked default(none) private(argc,b) firstprivate(argv) if(parallel: argc > 0) num_threads(ee) copyin(a) proc_bind(spread) reduction(|: c,d,arr1[argc]) reduction(*: e,arr[:10][0:argc]) allocate(e)
  foo();
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp parallel masked allocate(e) if(b) num_threads(c) proc_bind(close) reduction(^: e,f) reduction(&&: g,arr[0:argc][:10])
// CHECK-NEXT: foo()
#pragma omp parallel masked allocate(e) if (b) num_threads(c) proc_bind(close) reduction(^:e, f) reduction(&& : g, arr[0:argc][:10])
  foo();
  return tmain<int, 5>(b, &b) + tmain<long, 1>(x, &x);
}

template<typename T>
T S<T>::TS = 0;

#endif
