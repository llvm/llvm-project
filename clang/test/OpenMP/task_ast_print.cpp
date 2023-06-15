// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

typedef void *omp_depend_t;
typedef unsigned long omp_event_handle_t;

void foo() {}

struct S1 {
  S1(): a(0) {}
  S1(int v) : a(v) {}
  int a;
  typedef int type;
  S1 operator +(const S1&);
};

template <typename T>
class S7 : public T {
protected:
  T a, b, c[10], d[10];
  S7() : a(0) {}

public:
  S7(typename T::type v) : a(v) {
    omp_depend_t x;
    omp_event_handle_t evt;
#pragma omp taskgroup allocate(b) task_reduction(+:b)
#pragma omp task private(a) private(this->a) private(T::a) in_reduction(+:this->b) allocate(b) depend(depobj:x) detach(evt) depend(iterator(i=0:10:1, T *k = &a:&b), in: c[i], d[(int)(k-&a)]) affinity(iterator(i=0:10:1, T *k = &a:&b): c[i], d[(int)(k-&a)])
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

// CHECK: #pragma omp taskgroup allocate(this->b) task_reduction(+: this->b)
// CHECK: #pragma omp task private(this->a) private(this->a) private(T::a) in_reduction(+: this->b) allocate(this->b) depend(depobj : x) detach(evt) depend(iterator(int i = 0:10:1, T * k = &this->a:&this->b), in : this->c[i],this->d[(int)(k - &this->a)]) affinity(iterator(int i = 0:10:1, T * k = &this->a:&this->b) : this->c[i],this->d[(int)(k - &this->a)]){{$}}
// CHECK: #pragma omp task private(this->a) private(this->a)
// CHECK: #pragma omp task private(this->a) private(this->a) private(this->S1::a) in_reduction(+: this->b) allocate(this->b) depend(depobj : x) detach(evt) depend(iterator(int i = 0:10:1, S1 * k = &this->a:&this->b), in : this->c[i],this->d[(int)(k - &this->a)]) affinity(iterator(int i = 0:10:1, S1 * k = &this->a:&this->b) : this->c[i],this->d[(int)(k - &this->a)])

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

template <typename T, int C>
T tmain(T argc, T *argv) {
  T b = argc, c, d, e, f, g;
  static T a;
  S<T> s;
  T arr[argc];
  omp_depend_t x;
  omp_event_handle_t evt;
  double *arr_double;
#pragma omp task untied depend(in : argc, argv[b:argc], arr[:], ([argc][sizeof(T)])argv, arr_double[argc]) if (task : argc > 0) depend(depobj: x) detach(evt)
  a = 2;
#pragma omp task default(none), private(argc, b) firstprivate(argv) shared(d) if (argc > 0) final(S<T>::TS > 0) priority(argc) affinity(argc, argv[b:argc], arr[:], ([argc][sizeof(T)])argv)
  foo();
#pragma omp taskgroup task_reduction(-: argc)
#pragma omp task if (C) mergeable priority(C) in_reduction(-: argc)
  foo();
  return 0;
}

// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T b = argc, c, d, e, f, g;
// CHECK-NEXT: static T a;
// CHECK-NEXT: S<T> s;
// CHECK-NEXT: T arr[argc];
// CHECK-NEXT: omp_depend_t x;
// CHECK-NEXT: omp_event_handle_t evt;
// CHECK-NEXT: double *arr_double;
// CHECK-NEXT: #pragma omp task untied depend(in : argc,argv[b:argc],arr[:],([argc][sizeof(T)])argv,arr_double[argc]) if(task: argc > 0) depend(depobj : x) detach(evt)
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp task default(none) private(argc,b) firstprivate(argv) shared(d) if(argc > 0) final(S<T>::TS > 0) priority(argc) affinity(argc,argv[b:argc],arr[:],([argc][sizeof(T)])argv)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp taskgroup task_reduction(-: argc)
// CHECK-NEXT: #pragma omp task if(C) mergeable priority(C) in_reduction(-: argc)
// CHECK-NEXT: foo()
// CHECK: template<> int tmain<int, 5>(int argc, int *argv) {
// CHECK-NEXT: int b = argc, c, d, e, f, g;
// CHECK-NEXT: static int a;
// CHECK-NEXT: S<int> s;
// CHECK-NEXT: int arr[argc];
// CHECK-NEXT: omp_depend_t x;
// CHECK-NEXT: omp_event_handle_t evt;
// CHECK-NEXT: double *arr_double;
// CHECK-NEXT: #pragma omp task untied depend(in : argc,argv[b:argc],arr[:],([argc][sizeof(int)])argv,arr_double[argc]) if(task: argc > 0) depend(depobj : x) detach(evt)
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp task default(none) private(argc,b) firstprivate(argv) shared(d) if(argc > 0) final(S<int>::TS > 0) priority(argc) affinity(argc,argv[b:argc],arr[:],([argc][sizeof(int)])argv)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp taskgroup task_reduction(-: argc)
// CHECK-NEXT: #pragma omp task if(5) mergeable priority(5) in_reduction(-: argc)
// CHECK-NEXT: foo()
// CHECK: template<> long tmain<long, 1>(long argc, long *argv) {
// CHECK-NEXT: long b = argc, c, d, e, f, g;
// CHECK-NEXT: static long a;
// CHECK-NEXT: S<long> s;
// CHECK-NEXT: long arr[argc];
// CHECK-NEXT: omp_depend_t x;
// CHECK-NEXT: omp_event_handle_t evt;
// CHECK-NEXT: double *arr_double;
// CHECK-NEXT: #pragma omp task untied depend(in : argc,argv[b:argc],arr[:],([argc][sizeof(long)])argv,arr_double[argc]) if(task: argc > 0) depend(depobj : x) detach(evt)
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp task default(none) private(argc,b) firstprivate(argv) shared(d) if(argc > 0) final(S<long>::TS > 0) priority(argc) affinity(argc,argv[b:argc],arr[:],([argc][sizeof(long)])argv)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp taskgroup task_reduction(-: argc)
// CHECK-NEXT: #pragma omp task if(1) mergeable priority(1) in_reduction(-: argc)
// CHECK-NEXT: foo()

enum Enum {};

int main(int argc, char **argv) {
  long x;
  int b = argc, c, d, e, f, g;
  static int a;
  int arr[10], arr1[argc];
  omp_depend_t y;
  omp_event_handle_t evt;
#pragma omp threadprivate(a)
  Enum ee;
// CHECK: Enum ee;
#pragma omp task untied mergeable depend(out:argv[:a][1], (arr)[0:],([argc][10])argv,b) if(task: argc > 0) priority(f) depend(depobj:y)
  // CHECK-NEXT: #pragma omp task untied mergeable depend(out : argv[:a][1],(arr)[0:],([argc][10])argv,b) if(task: argc > 0) priority(f) depend(depobj : y)
  a = 2;
// CHECK-NEXT: a = 2;
#pragma omp taskgroup task_reduction(min: arr1)
#pragma omp task default(none), private(argc, b) firstprivate(argv, evt) if (argc > 0) final(a > 0) depend(inout : a, argv[:argc],arr[:a], ([10][argc])argv) priority(23) in_reduction(min: arr1), detach(evt)
  // CHECK-NEXT: #pragma omp taskgroup task_reduction(min: arr1)
  // CHECK-NEXT: #pragma omp task default(none) private(argc,b) firstprivate(argv,evt) if(argc > 0) final(a > 0) depend(inout : a,argv[:argc],arr[:a],([10][argc])argv) priority(23) in_reduction(min: arr1) detach(evt)
  foo();
  // CHECK-NEXT: foo();
#pragma omp taskgroup task_reduction(min: arr1)
#pragma omp parallel reduction(+:arr1)
#pragma omp task in_reduction(min: arr1) depend(iterator(i=0:argc, unsigned j=argc:0:a), out: argv[i][j])
  // CHECK-NEXT: #pragma omp taskgroup task_reduction(min: arr1)
  // CHECK-NEXT: #pragma omp parallel reduction(+: arr1)
  // CHECK-NEXT: #pragma omp task in_reduction(min: arr1) depend(iterator(int i = 0:argc, unsigned int j = argc:0:a), out : argv[i][j])
  foo();
  // CHECK-NEXT: foo();
  // CHECK-NEXT: #pragma omp task in_reduction(+: arr1)
#pragma omp task in_reduction(+: arr1)
  foo();
  // CHECK-NEXT: foo();
  // CHECK-NEXT: #pragma omp task depend(out : arr,omp_all_memory)
#pragma omp task depend(out: omp_all_memory, arr)
  foo();
  // CHECK-NEXT: foo();
  // CHECK-NEXT: #pragma omp task depend(inout : b,arr,a,x,omp_all_memory)
#pragma omp task depend(inout: b, arr, omp_all_memory, a, x)
  foo();
  // CHECK-NEXT: foo();
  // CHECK-NEXT: #pragma omp task depend(inout : omp_all_memory)
#pragma omp task depend(inout: omp_all_memory)
  foo();
  // CHECK-NEXT: foo();
  return tmain<int, 5>(b, &b) + tmain<long, 1>(x, &x);
}

extern template int S<int>::TS;
extern template long S<long>::TS;

#endif
