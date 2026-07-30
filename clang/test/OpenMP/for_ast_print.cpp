// RUN: %clang_cc1 -verify -fopenmp -ast-print %s -Wsign-conversion | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -ast-print %s -Wsign-conversion -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s -DOMP51
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -std=c++11 -include-pch %t -verify %s -ast-print -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -ast-print %s -Wsign-conversion -DOMP52 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP52
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -x c++ -std=c++11 -emit-pch -o %t %s -DOMP52
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -std=c++11 -include-pch %t -verify %s -ast-print -DOMP52 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP52

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s -Wsign-conversion | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 -ast-print %s -Wsign-conversion -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s -DOMP51
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -std=c++11 -include-pch %t -verify %s -ast-print -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=52 -ast-print %s -Wsign-conversion -DOMP52 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP52
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=52 -x c++ -std=c++11 -emit-pch -o %t %s -DOMP52
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=52 -std=c++11 -include-pch %t -verify %s -ast-print -DOMP52 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP52
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
#if defined(OMP52)
#pragma omp for linear(c: val)
#else
#pragma omp for linear(val(c))
#endif
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
#if defined(OMP52)
#pragma omp for linear(this->b: uval)
#else
#pragma omp for linear(uval(this->b))
#endif
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp for private(this->a) private(this->a) private(T::a)
// CHECK: #pragma omp for lastprivate(this->a) lastprivate(this->a) lastprivate(T::a)
// CHECK: #pragma omp for linear(this->c: val)
// CHECK: #pragma omp for private(this->a) private(this->a)
// CHECK: #pragma omp for lastprivate(this->a) lastprivate(this->a)
// CHECK: #pragma omp for linear(this->b: uval)
// CHECK: #pragma omp for private(this->a) private(this->a) private(this->S::a)
// CHECK: #pragma omp for lastprivate(this->a) lastprivate(this->a) lastprivate(this->S::a)
// CHECK: #pragma omp for linear(this->c: val)

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
#if defined(OMP52)
#pragma omp for linear(S7<S>::d: ref)
#else
#pragma omp for linear(ref(S7<S>::d))
#endif
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
// CHECK: #pragma omp for linear(this->S7<S>::d: ref)
// CHECK: #pragma omp for private(this->a) private(this->a)
// CHECK: #pragma omp for lastprivate(this->a) lastprivate(this->a)
// CHECK: #pragma omp for linear(this->c)

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, g;
  T arr[N];
  static T a;
// CHECK: static T a;
#pragma omp for schedule(dynamic) linear(a) allocate(a)
  // CHECK-NEXT: #pragma omp for schedule(dynamic) linear(a) allocate(a)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp for schedule(nonmonotonic: dynamic)
  // CHECK-NEXT: #pragma omp for schedule(nonmonotonic: dynamic)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp for schedule(monotonic : dynamic)
  // CHECK-NEXT: #pragma omp for schedule(monotonic: dynamic)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp for schedule(nonmonotonic : guided)
  // CHECK-NEXT: #pragma omp for schedule(nonmonotonic: guided)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp for schedule(monotonic : guided)
  // CHECK-NEXT: #pragma omp for schedule(monotonic: guided)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp for schedule(nonmonotonic : static)
  // CHECK-NEXT: #pragma omp for schedule(nonmonotonic: static)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp for schedule(monotonic :static)
  // CHECK-NEXT: #pragma omp for schedule(monotonic: static)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp for schedule(nonmonotonic : auto)
  // CHECK-NEXT: #pragma omp for schedule(nonmonotonic: auto)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp for schedule(monotonic : auto)
  // CHECK-NEXT: #pragma omp for schedule(monotonic: auto)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp for schedule(nonmonotonic : runtime)
  // CHECK-NEXT: #pragma omp for schedule(nonmonotonic: runtime)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp for schedule(monotonic : runtime)
  // CHECK-NEXT: #pragma omp for schedule(monotonic: runtime)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp parallel
#pragma omp for allocate(argc) private(argc, b), firstprivate(c, d), lastprivate(d, f) collapse(N) schedule(static, N) ordered(N) nowait
  for (auto &x : arr) {
    int j, hhh = 0;
    for (int i = 0; i < 2; ++i) {
      int j, hhh = 0;
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
      ++hhh;
    }
    ++hhh;
  }
  // CHECK-NEXT: #pragma omp parallel
  // CHECK-NEXT: #pragma omp for allocate(argc) private(argc,b) firstprivate(c,d) lastprivate(d,f) collapse(N) schedule(static, N) ordered(N) nowait
  // CHECK-NEXT: for (auto &x : arr) {
  // CHECK-NEXT: int j, hhh = 0;
  // CHECK-NEXT: for (int i = 0; i < 2; ++i) {
  // CHECK-NEXT: int j, hhh = 0;
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
  // CHECK-NEXT: ++hhh;
  // CHECK-NEXT: }
  // CHECK-NEXT: ++hhh;
  // CHECK-NEXT: }
  return T();
}

int main(int argc, char **argv) {
// CHECK: int main(int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  float arr[20];
  static int a;
// CHECK: static int a;
#ifdef OMP52
#pragma omp for schedule(guided, argc) reduction(+:argv[0][:1]) order(unconstrained:concurrent)
#elif OMP51
#pragma omp for schedule(guided, argc) reduction(+:argv[0][:1]) order(unconstrained:concurrent)
#else
#pragma omp for schedule(guided, argc) reduction(+:argv[0][:1]) order(concurrent)
#endif // OMP51
  // OMP50: #pragma omp for schedule(guided, argc) reduction(+: argv[0][:1]) order(concurrent)
  // OMP51: #pragma omp for schedule(guided, argc) reduction(+: argv[0][:1]) order(unconstrained: concurrent)
  // OMP52: #pragma omp for schedule(guided, argc) reduction(+: argv[0][:1]) order(unconstrained: concurrent)
  for (int i = argc; i < c; ++i)
    a = 2;
// CHECK-NEXT: for (int i = argc; i < c; ++i)
// CHECK-NEXT: a = 2;
#pragma omp parallel
#ifdef OMP52 
#pragma omp for private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(3) schedule(auto) ordered nowait linear(g: step(-1)) reduction(task, +:e)
#else
#pragma omp for private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(3) schedule(auto) ordered nowait linear(g:-1) reduction(task, +:e)
#endif
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      for (auto x : arr)
        foo(), (void)x;
  // CHECK-NEXT: #pragma omp parallel
  // CHECK-NEXT: #pragma omp for private(argc,b) firstprivate(argv,c) lastprivate(d,f) collapse(3) schedule(auto) ordered nowait linear(g: step(-1)) reduction(task, +: e)
  // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: for (int j = 0; j < 10; ++j)
  // CHECK-NEXT: for (auto x : arr)
  // CHECK-NEXT: foo() , (void)x;
  char buf[9] = "01234567";
  char *p, *q;
#pragma omp parallel
#pragma omp for
  for (p = buf; p < &buf[8]; p++)
    for (q = &buf[0]; q <= buf + 7; q++)
      foo();
  // CHECK: #pragma omp parallel
  // CHECK-NEXT: #pragma omp for{{$}}
  // CHECK-NEXT: for (p = buf; p < &buf[8]; p++)
  // CHECK-NEXT: for (q = &buf[0]; q <= buf + 7; q++)
  // CHECK-NEXT: foo();
  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
