// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s
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
#pragma omp taskgroup task_reduction(+ : a) task_reduction(*: b[:])
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
#pragma omp taskgroup task_reduction(&& : this->a) task_reduction(^: b[s.a.a])
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp taskgroup task_reduction(+: this->a) task_reduction(*: this->b[:])
// CHECK: #pragma omp taskgroup task_reduction(&&: this->a) task_reduction(^: this->b[s.a.a])
// CHECK: #pragma omp taskgroup task_reduction(+: this->a) task_reduction(*: this->b[:])

class S8 : public S7<S1> {
  S8() {}

public:
  S8(int v) : S7<S1>(v){
#pragma omp taskgroup task_reduction(^ : S7 < S1 > ::a) task_reduction(+ : S7 < S1 > ::b[ : S7 < S1 > ::a.a])
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S8 &operator=(S8 &s) {
#pragma omp taskgroup task_reduction(* : this->a) task_reduction(&&:this->b[a.a:])
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp taskgroup task_reduction(^: this->S7<S1>::a) task_reduction(+: this->S7<S1>::b[:this->S7<S1>::a.a])
// CHECK: #pragma omp taskgroup task_reduction(*: this->a) task_reduction(&&: this->b[this->a.a:])

int main (int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  static int a;
// CHECK: static int a;
#pragma omp taskgroup
  a=2;
// CHECK-NEXT: #pragma omp taskgroup{{$}}
// CHECK-NEXT: a = 2;
// CHECK-NEXT: ++a;
  ++a;
#pragma omp taskgroup task_reduction(min: a)
  foo();
// CHECK-NEXT: #pragma omp taskgroup task_reduction(min: a)
// CHECK-NEXT: foo();
// CHECK-NEXT: return 0;
  return 0;
}

#endif
