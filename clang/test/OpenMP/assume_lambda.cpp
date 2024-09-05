// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -ast-print %s | FileCheck %s
// expected-no-diagnostics

extern int bar(int);

int foo(int arg)
{
  #pragma omp assume no_openmp_routines
  {
    auto fn = [](int x) { return bar(x); };
// CHECK: auto fn = [](int x) {
    return fn(5);
  }
}

class C {
public:
  int foo(int a);
};

// We're really just checking that this parses.  All the assumptions are thrown
// away immediately for now.
int C::foo(int a)
{
  #pragma omp assume holds(sizeof(void*) == 8) absent(parallel)
  {
    auto fn = [](int x) { return bar(x); };
// CHECK: auto fn = [](int x) {
    return fn(5);
  }
}
