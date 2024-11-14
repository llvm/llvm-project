// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -ast-print %s -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s -DOMP51
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -std=c++11 -include-pch %t -verify %s -ast-print -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -ast-print %s -DOMP52 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP52
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -x c++ -std=c++11 -emit-pch -o %t %s -DOMP52
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -std=c++11 -include-pch %t -verify %s -ast-print -DOMP52 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP52

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 -ast-print %s -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s -DOMP51
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -std=c++11 -include-pch %t -verify %s -ast-print -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// expected-no-diagnostics
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=52 -ast-print %s -DOMP52 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP52
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
  S7() : a(0) {}

public:
  S7(typename T::type v) : a(v) {
#pragma omp parallel for private(a) private(this->a) private(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
#pragma omp parallel for private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp parallel for private(this->a) private(this->a) private(T::a){{$}}
// CHECK: #pragma omp parallel for private(this->a) private(this->a)
// CHECK: #pragma omp parallel for private(this->a) private(this->a) private(this->S::a)

class S8 : public S7<S> {
  S8() {}

public:
  S8(int v) : S7<S>(v){
#pragma omp parallel for private(a) private(this->a) private(S7<S>::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S8 &operator=(S8 &s) {
#pragma omp parallel for private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp parallel for private(this->a) private(this->a) private(this->S7<S>::a)
// CHECK: #pragma omp parallel for private(this->a) private(this->a)

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, h;
  static T a;
// CHECK: static T a;
  static T g;
#pragma omp threadprivate(g)
#if defined(OMP51) || defined(OMP52)
#pragma omp parallel for schedule(dynamic) default(none) copyin(g) linear(a) allocate(a) lastprivate(conditional: d, e,f) order(reproducible:concurrent)
#else
#pragma omp parallel for schedule(dynamic) default(none) copyin(g) linear(a) allocate(a) lastprivate(conditional: d, e,f) order(concurrent)
#endif // OMP51
  // OMP50: #pragma omp parallel for schedule(dynamic) default(none) copyin(g) linear(a) allocate(a) lastprivate(conditional: d,e,f) order(concurrent)
  // OMP51: #pragma omp parallel for schedule(dynamic) default(none) copyin(g) linear(a) allocate(a) lastprivate(conditional: d,e,f) order(reproducible: concurrent)
  // OMP52: #pragma omp parallel for schedule(dynamic) default(none) copyin(g) linear(a) allocate(a) lastprivate(conditional: d,e,f) order(reproducible: concurrent)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp parallel for allocate(argc) private(argc, b), firstprivate(c, d), lastprivate(d, f) collapse(N) schedule(static, N) ordered(N) if (parallel :argc) num_threads(N) default(shared) shared(e) reduction(+ : h)
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
  // CHECK-NEXT: #pragma omp parallel for allocate(argc) private(argc,b) firstprivate(c,d) lastprivate(d,f) collapse(N) schedule(static, N) ordered(N) if(parallel: argc) num_threads(N) default(shared) shared(e) reduction(+: h)
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

int increment () {
  #pragma omp for
  for (int i = 5 ; i != 0; ++i)
    ;
  // CHECK:      int increment() {
  // CHECK-NEXT:   #pragma omp for
  // CHECK-NEXT:     for (int i = 5; i != 0; ++i)
  // CHECK-NEXT:       ;
  return 0;
}

int decrement_nowait () {
  #pragma omp for nowait
  for (int j = 5 ; j != 0; --j)
    ;
  // CHECK:      int decrement_nowait() {
  // CHECK-NEXT:   #pragma omp for nowait
  // CHECK-NEXT:     for (int j = 5; j != 0; --j)
  // CHECK-NEXT:       ;
  return 0;
}


int main(int argc, char **argv) {
  int b = argc, c, d, e, f, h;
  static int a;
// CHECK: static int a;
  static float g;
#pragma omp threadprivate(g)
#pragma omp parallel for schedule(guided, argc) default(none) copyin(g) linear(a) shared(argc) reduction(task,&:d)
  // CHECK: #pragma omp parallel for schedule(guided, argc) default(none) copyin(g) linear(a) shared(argc) reduction(task, &: d)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#ifdef OMP52
#pragma omp parallel for private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(2) schedule(auto) ordered if (argc) num_threads(a) default(shared) shared(e) reduction(+ : h) linear(a: step(-5))
#else
#pragma omp parallel for private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(2) schedule(auto) ordered if (argc) num_threads(a) default(shared) shared(e) reduction(+ : h) linear(a:-5)
#endif
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      foo();
  // CHECK-NEXT: #pragma omp parallel for private(argc,b) firstprivate(argv,c) lastprivate(d,f) collapse(2) schedule(auto) ordered if(argc) num_threads(a) default(shared) shared(e) reduction(+: h) linear(a: step(-5))
 // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: for (int j = 0; j < 10; ++j)
  // CHECK-NEXT: foo();
  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
