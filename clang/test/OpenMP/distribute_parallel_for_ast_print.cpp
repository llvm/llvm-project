// RUN: %clang_cc1 -verify -std=c++11 -fopenmp -fopenmp-version=45 -ast-print %s -Wno-openmp-mapping | FileCheck %s --check-prefix=CHECK --check-prefix=OMP45
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -include-pch %t -verify %s -ast-print -Wno-openmp-mapping | FileCheck %s --check-prefix=CHECK --check-prefix=OMP45
// RUN: %clang_cc1 -verify -std=c++11 -fopenmp -ast-print %s -Wno-openmp-mapping -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s -DOMP5
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -verify %s -ast-print -Wno-openmp-mapping -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -verify -std=c++11 -fopenmp -fopenmp-version=51 -ast-print %s -Wno-openmp-mapping -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s -DOMP51
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -std=c++11 -include-pch %t -verify %s -ast-print -Wno-openmp-mapping -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51

// RUN: %clang_cc1 -verify -std=c++11 -fopenmp-simd -fopenmp-version=45 -ast-print %s -Wno-openmp-mapping | FileCheck %s --check-prefix=CHECK --check-prefix=OMP45
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -std=c++11 -include-pch %t -verify %s -ast-print -Wno-openmp-mapping | FileCheck %s --check-prefix=CHECK --check-prefix=OMP45
// RUN: %clang_cc1 -verify -std=c++11 -fopenmp-simd -ast-print %s -Wno-openmp-mapping -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s -DOMP5
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -verify %s -ast-print -Wno-openmp-mapping -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -verify -std=c++11 -fopenmp-simd -fopenmp-version=51 -ast-print %s -Wno-openmp-mapping -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s -DOMP51
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -std=c++11 -include-pch %t -verify %s -ast-print -Wno-openmp-mapping -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

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
#pragma omp teams
#pragma omp distribute parallel for private(a) private(this->a) private(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp distribute parallel for private(this->a) private(this->a) private(T::a){{$}}
// CHECK: #pragma omp distribute parallel for private(this->a) private(this->a)
// CHECK: #pragma omp distribute parallel for private(this->a) private(this->a) private(this->S::a)

class S8 : public S7<S> {
  S8() {}

public:
  S8(int v) : S7<S>(v){
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for private(a) private(this->a) private(S7<S>::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S8 &operator=(S8 &s) {
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp distribute parallel for private(this->a) private(this->a) private(this->S7<S>::a)
// CHECK: #pragma omp distribute parallel for private(this->a) private(this->a)

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, h;
  static T a;
// CHECK: static T a;
  static T g;
#pragma omp threadprivate(g)
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for dist_schedule(static, a) schedule(dynamic) default(none) copyin(g) firstprivate(a) allocate(a)
  // CHECK: #pragma omp distribute parallel for dist_schedule(static, a) schedule(dynamic) default(none) copyin(g) firstprivate(a) allocate(a)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for allocate(argc) private(argc, b), firstprivate(c, d), lastprivate(f) collapse(N) schedule(static, N) if (parallel :argc) num_threads(N) default(shared) shared(e) reduction(+ : h) dist_schedule(static,N)
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
            a++;
  // CHECK: #pragma omp distribute parallel for allocate(argc) private(argc,b) firstprivate(c,d) lastprivate(f) collapse(N) schedule(static, N) if(parallel: argc) num_threads(N) default(shared) shared(e) reduction(+: h) dist_schedule(static, N)
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
  // CHECK-NEXT: a++;
  return T();
}

void foo(int argc, char **argv) {
  int b, c, d, e, f, h;
  static int a;
// CHECK: static int a;
  static float g;
#pragma omp threadprivate(g)
  [&]() {
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for schedule(guided, argc) default(none) copyin(g) dist_schedule(static, a) private(a) shared(argc)
    // CHECK: #pragma omp distribute parallel for schedule(guided, argc) default(none) copyin(g) dist_schedule(static, a) private(a) shared(argc)
    for (int i = 0; i < 2; ++i)
      // CHECK: for (int i = 0; i < 2; ++i)
      [&]() {
        a = 2;
        // CHECK: a = 2;
      }();

  }();
  [&]() {
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(2) schedule(auto) if (argc) num_threads(a) default(shared) shared(e) reduction(+ : h) dist_schedule(static, b)
    for (int i = 0; i < 10; ++i)
      for (int j = 0; j < 10; ++j)
        [&]() {
          a++;
        }();
    // CHECK: #pragma omp distribute parallel for private(argc,b) firstprivate(argv,c) lastprivate(d,f) collapse(2) schedule(auto) if(argc) num_threads(a) default(shared) shared(e) reduction(+: h) dist_schedule(static, b)
    // CHECK-NEXT: for (int i = 0; i < 10; ++i)
    // CHECK-NEXT: for (int j = 0; j < 10; ++j)
    // CHECK: a++;
  }();
}

int main(int argc, char **argv) {
  int b = argc, c, d, e, f, h;
  static int a;
// CHECK: static int a;
  static float g;
#pragma omp threadprivate(g)
#pragma omp target
#pragma omp teams
#ifdef OMP51
#pragma omp distribute parallel for schedule(guided, argc) default(none) copyin(g) dist_schedule(static, a) private(a) shared(argc) order(reproducible:concurrent) reduction(task,+:c)
#elif OMP5
#pragma omp distribute parallel for schedule(guided, argc) default(none) copyin(g) dist_schedule(static, a) private(a) shared(argc) order(concurrent) reduction(task,+:c)
#else
#pragma omp distribute parallel for schedule(guided, argc) default(none) copyin(g) dist_schedule(static, a) private(a) shared(argc)
#endif // OMP5
  // OMP51: #pragma omp distribute parallel for schedule(guided, argc) default(none) copyin(g) dist_schedule(static, a) private(a) shared(argc) order(reproducible: concurrent) reduction(task, +: c)
  // OMP50: #pragma omp distribute parallel for schedule(guided, argc) default(none) copyin(g) dist_schedule(static, a) private(a) shared(argc) order(concurrent) reduction(task, +: c)
  // OMP45: #pragma omp distribute parallel for schedule(guided, argc) default(none) copyin(g) dist_schedule(static, a) private(a) shared(argc)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(2) schedule(auto) if (argc) num_threads(a) default(shared) shared(e) reduction(+ : h) dist_schedule(static, b)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      a++;
  // CHECK: #pragma omp distribute parallel for private(argc,b) firstprivate(argv,c) lastprivate(d,f) collapse(2) schedule(auto) if(argc) num_threads(a) default(shared) shared(e) reduction(+: h) dist_schedule(static, b)
 // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: for (int j = 0; j < 10; ++j)
  // CHECK-NEXT: a++;
  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
