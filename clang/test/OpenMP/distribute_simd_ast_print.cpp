// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -ast-print %s -Wno-openmp-mapping | FileCheck %s --check-prefix=CHECK --check-prefix=OMP45
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print -Wno-openmp-mapping | FileCheck %s --check-prefix=CHECK --check-prefix=OMP45
// RUN: %clang_cc1 -verify -fopenmp -ast-print %s -Wno-openmp-mapping -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s -DOMP5
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print -Wno-openmp-mapping -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -ast-print %s -Wno-openmp-mapping -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s -DOMP51
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print -Wno-openmp-mapping -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -ast-print %s -Wno-openmp-mapping | FileCheck %s --check-prefix=CHECK --check-prefix=OMP45
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print -Wno-openmp-mapping | FileCheck %s --check-prefix=CHECK --check-prefix=OMP45
// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s -Wno-openmp-mapping -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s -DOMP5
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print -Wno-openmp-mapping -DOMP5 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP50
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 -ast-print %s -Wno-openmp-mapping -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s -DOMP51
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print -Wno-openmp-mapping -DOMP51 | FileCheck %s --check-prefix=CHECK --check-prefix=OMP51
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
#pragma omp distribute simd private(a) private(this->a) private(T::a) allocate(T::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S7 &operator=(S7 &s) {
#pragma omp target
#pragma omp teams
#pragma omp distribute simd allocate(a) private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp distribute simd private(this->a) private(this->a) private(T::a) allocate(T::a){{$}}
// CHECK: #pragma omp distribute simd allocate(this->a) private(this->a) private(this->a)
// CHECK: #pragma omp distribute simd private(this->a) private(this->a) private(this->S::a) allocate(this->S::a)

class S8 : public S7<S> {
  S8() {}

public:
  S8(int v) : S7<S>(v){
#pragma omp target
#pragma omp teams
#pragma omp distribute simd private(a) private(this->a) private(S7<S>::a)
    for (int k = 0; k < a.a; ++k)
      ++this->a.a;
  }
  S8 &operator=(S8 &s) {
#pragma omp target
#pragma omp teams
#pragma omp distribute simd private(a) private(this->a)
    for (int k = 0; k < s.a.a; ++k)
      ++s.a.a;
    return *this;
  }
};

// CHECK: #pragma omp distribute simd private(this->a) private(this->a) private(this->S7<S>::a)
// CHECK: #pragma omp distribute simd private(this->a) private(this->a)

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, h;
  static T a;
// CHECK: static T a;
  static T g;
#pragma omp threadprivate(g)

#pragma omp target
#pragma omp teams
#pragma omp distribute simd dist_schedule(static, a) firstprivate(a)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK: #pragma omp distribute simd dist_schedule(static, a) firstprivate(a)
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;

#pragma omp target
#pragma omp teams
#pragma omp distribute simd private(argc, b), firstprivate(c, d), lastprivate(f) collapse(N) reduction(+ : h) dist_schedule(static,N)
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 10; ++k)
        for (int m = 0; m < 10; ++m)
          for (int n = 0; n < 10; ++n)
            a++;
// CHECK: #pragma omp distribute simd private(argc,b) firstprivate(c,d) lastprivate(f) collapse(N) reduction(+: h) dist_schedule(static, N)
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: for (int j = 0; j < 2; ++j)
// CHECK-NEXT: for (int k = 0; k < 10; ++k)
// CHECK-NEXT: for (int m = 0; m < 10; ++m)
// CHECK-NEXT: for (int n = 0; n < 10; ++n)
// CHECK-NEXT: a++;
  return T();
}

int main(int argc, char **argv) {
  int b = argc, c, d, e, f, h;
  int x[200];
  static int a;
// CHECK: static int a;
  static float g;
#pragma omp threadprivate(g)

#pragma omp target
#pragma omp teams
#pragma omp distribute simd dist_schedule(static, a) private(a)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK: #pragma omp distribute simd  dist_schedule(static, a) private(a)
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;

#pragma omp target
#pragma omp teams
#ifdef OMP5
#pragma omp distribute simd private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(2) reduction(+ : h) dist_schedule(static, b) if(simd:argc)
#else
#pragma omp distribute simd private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(2) reduction(+ : h) dist_schedule(static, b)
#endif // OMP5
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
            a++;
// OMP45: #pragma omp distribute simd private(argc,b) firstprivate(argv,c) lastprivate(d,f) collapse(2) reduction(+: h) dist_schedule(static, b)
// OMP50: #pragma omp distribute simd private(argc,b) firstprivate(argv,c) lastprivate(d,f) collapse(2) reduction(+: h) dist_schedule(static, b) if(simd: argc)
// OMP51: #pragma omp distribute simd private(argc,b) firstprivate(argv,c) lastprivate(d,f) collapse(2) reduction(+: h) dist_schedule(static, b)
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: for (int j = 0; j < 10; ++j)
// CHECK-NEXT: a++;

  int i;
#pragma omp target
#pragma omp teams
#ifdef OMP51
#pragma omp distribute simd aligned(x:8) linear(i:2) safelen(8) simdlen(8) if(argc) nontemporal(argc, c, d) order(reproducible:concurrent)
#elif OMP5
#pragma omp distribute simd aligned(x:8) linear(i:2) safelen(8) simdlen(8) if(argc) nontemporal(argc, c, d) order(concurrent)
#else
#pragma omp distribute simd aligned(x:8) linear(i:2) safelen(8) simdlen(8)
#endif // OMP51
  for (i = 0; i < 100; i++)
    for (int j = 0; j < 200; j++)
      a += h + x[j];
// OMP45: #pragma omp distribute simd aligned(x: 8) linear(i: 2) safelen(8) simdlen(8)
// OMP50: #pragma omp distribute simd aligned(x: 8) linear(i: 2) safelen(8) simdlen(8) if(argc) nontemporal(argc,c,d) order(concurrent)
// OMP51: #pragma omp distribute simd aligned(x: 8) linear(i: 2) safelen(8) simdlen(8) if(argc) nontemporal(argc,c,d) order(reproducible: concurrent)
// CHECK-NEXT: for (i = 0; i < 100; i++)
// CHECK-NEXT: for (int j = 0; j < 200; j++)
// CHECK-NEXT: a += h + x[j];

  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
