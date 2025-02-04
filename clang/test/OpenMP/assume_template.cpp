// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

extern int qux(int);

template<typename T>
int foo(T arg)
{
  #pragma omp assume no_openmp_routines
  {
    auto fn = [](int x) { return qux(x); };
// CHECK: auto fn = [](int x) {
    return fn(5);
  }
}

template<typename T>
class C {
  T m;

public:
  T bar(T a);
};

// We're really just checking this parses.  All the assumptions are thrown
// away immediately for now.
template<typename T>
T C<T>::bar(T a)
{
  #pragma omp assume holds(sizeof(T) == 8) absent(parallel)
  {
    return (T)qux((int)a);
// CHECK: return (T)qux((int)a);
  }
}

#endif
